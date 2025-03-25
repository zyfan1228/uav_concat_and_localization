import os
import math
import time
from os.path import join
from PIL import Image
import gc

import numpy as np
import pandas as pd
import imutils
import cv2
import piexif
# import pyvips
from fractions import Fraction

from get_info import save_csv


CONSTANTS_PIXEL_PITCH = 2.55 # 实验结果为：M3TD的广角相机像元距离约为 2.5448 微米 
# CONSTANTS_PIXEL_PITCH = 1.6 # 实验结果为：M30T的广角相机像元距离约为 1.6 微米 



def compute_m_per_pixel(f_length, height, p=2.5448) -> float:
    '''
        args:
            p: 像元大小(μm)
            f_length: 焦距(mm);
            height: 无人机飞行相对地面高度(m) (非海拔高度);
        return:
            返回每个像素代表的实际长度(m)
    '''
    return p * height / f_length * 1e-3

def gps2xy(lat, lon, ref_lat, ref_lon):
    """
    使用 Haversine 公式计算目标点与参考点之间的相对坐标。
    :param lat: 目标点的纬度
    :param lon: 目标点的经度
    :param ref_lat: 参考点的纬度
    :param ref_lon: 参考点的经度
    :return: 相对位置坐标 (x, y)，单位：米
    """
    a = 6378137.0  # 赤道半径 (米)
    b = 6356752.3142  # 极半径 (米)
    e = math.sqrt(1 - (b / a) ** 2)  # 偏心率

    # 目标点相对于参考点的纬度差和经度差
    dlat_deg = lat - ref_lat  # 纬度差
    dlon_deg = lon - ref_lon  # 经度差

    # 转换为弧度
    phi = math.radians(ref_lat)

    # 计算ref_lat附近的1度纬度的长度（单位：米）
    meters_per_degree_lat = (math.pi / 180) * a * (1 - e**2) / \
        (1 - e**2 * math.sin(phi)**2)**(3/2) * 0.95

    # 计算ref_lat附近的1度经度的长度（单位：米）
    meters_per_degree_lon = (math.pi / 180) * a * math.cos(phi) / \
        math.sqrt(1 - e**2 * math.sin(phi)**2) * 0.975

    # 计算 X, Y 坐标
    y = dlon_deg * meters_per_degree_lon  # 东向坐标（正东为正）
    x = dlat_deg * meters_per_degree_lat  # 北向坐标（正北为正）

    return x, y, [meters_per_degree_lat, meters_per_degree_lon]

def find_left_top(img_array, center_x, center_y):
    '''
        传入的center_x和center_y是正北方向为x正方向 正东方向为y正方向
    '''
    h, w, _ = img_array.shape

    return int(center_x - 0.5 * h), int(center_y - 0.5 * w)

def rotate_image(image, image_yaw):
    # 传入img和yaw角度，返回地理正确旋转之后的图片(上北下南)
    rotate_angle = image_yaw + 180
    image_rotated = imutils.rotate_bound(image, rotate_angle)

    return image_rotated

def rotate_and_smooth_edges(image, angle):
    # 旋转图像，保持图像完整
    rotated_image = imutils.rotate_bound(image, angle + 180)
    
    # 转换为灰度图像
    gray_image = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY)
    
    # 创建掩膜，检测接近黑色的区域
    _, mask = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY)  # threshold 参数可以调整
    
    # 反转掩膜，黑色区域变为 255，其他区域变为 0
    mask_inv = cv2.bitwise_not(mask)

    # 使用双边滤波或高斯模糊平滑边缘
    smoothed_image = cv2.bilateralFilter(rotated_image, d=5, sigmaColor=75, sigmaSpace=75)
    
    # 将平滑处理后的区域与原图的其他区域合并
    result = cv2.bitwise_and(rotated_image, rotated_image, mask=mask_inv)  # 保留非黑色区域
    smoothed_part = cv2.bitwise_and(smoothed_image, smoothed_image, mask=mask)  # 平滑处理黑色区域
    
    # 合并图像
    final_result = cv2.add(result, smoothed_part)
    
    return final_result

def decimal_to_dms(decimal):
    # 转换浮点经纬度为度分秒格式
    degree = int(decimal)
    minute = int((decimal - degree) * 60)
    second = ((decimal - degree) * 60 - minute) * 60

    fraction = Fraction(second).limit_denominator(1000000)
    return [(degree, 1), (minute, 1), (fraction.numerator, fraction.denominator)]

def main(imgs_path: str, save_path: str) -> list:
    """
        args:
            imgs_path: 航线数据文件夹(例如: "./DJI_202408051624_001")
            save_path: .csv文件和拼接后图片的保存的文件夹路径
        return:
            拼接后大图五点经纬度列表(纬度, 经度), 顺序依次为“中心点; 左下; 左上; 右下; 右上”
    """

    five_points_list = []

    # 航线数据文件夹path
    dir_path = imgs_path
    dir_name = os.path.split(dir_path)[-1]
    imgs_list = os.listdir(dir_path)
    # print(f"Total Images: {len(imgs_list)}")

    # 导出csv文件
    save_csv(dir_path, save_path)

    # 导入航线数据导出的csv文件, 会保存为"数据文件夹名_info.csv""
    csv_path = join(save_path, f"{dir_name}_info.csv")
    data = pd.read_csv(csv_path)

    la_list = data['la_decimal']
    lg_list = data['lg_decimal']
    rel_altitude_list = data['rel_altitude']
    f_length = data['focalLength'][0]
    yaw_list = data['yaw']

    # 找出无人机位置参考点（最左下，即最西南的点 + 最右上，即最东北的点）
    left_bottom_la = min(la_list)
    left_bottom_lg = min(lg_list)
    right_top_la = max(la_list)
    right_top_lg = max(lg_list)
    # 计算航线覆盖范围中心经纬度
    center_la = (left_bottom_la + right_top_la) / 2.0
    center_lg = (left_bottom_lg + right_top_lg) / 2.0
    five_points_list.append((center_la, center_lg))

    # 平均相对高度
    avg_height = np.mean(rel_altitude_list)

    # 每像素多少m
    m_per_pixel = compute_m_per_pixel(f_length, avg_height)

    # 计算左下与右上点间的实际距离和相对像素距离
    dx_m, dy_m, meters_per_du_list = gps2xy(right_top_la, right_top_lg, left_bottom_la, left_bottom_lg)
    dx_p, dy_p = dx_m / m_per_pixel, dy_m / m_per_pixel

    # 每像素多少度
    du_per_pixel_x = (right_top_la - left_bottom_la) / dx_p
    du_per_pixel_y = (right_top_lg - left_bottom_lg) / dy_p

    # 大图左下经纬度
    lb_la = left_bottom_la - 3024 * du_per_pixel_x
    lb_lg = left_bottom_lg - 4032 * du_per_pixel_y
    five_points_list.append((lb_la, lb_lg))

    concat_img_array = np.zeros((int(dx_p) + 2 * 3024, int(dy_p) + 2 * 4032, 3), 
                                dtype=np.uint8)
    
    huge_h, huge_w, _ = concat_img_array.shape

    # 大图左上经纬度
    lt_la = lb_la + huge_h * du_per_pixel_x
    lt_lg = lb_lg
    five_points_list.append((lt_la, lt_lg))
    # 大图右下经纬度
    rb_la = lb_la
    rb_lg = lb_lg + huge_w * du_per_pixel_y
    five_points_list.append((rb_la, rb_lg))
    # 大图右上经纬度
    rt_la = lt_la
    rt_lg = rb_lg
    five_points_list.append((rt_la, rt_lg))

    # 保存小图s中心坐标，北和东分别为x，y正方向
    coordinates_list = []
    # 保存小图s左上角坐标
    coordinates_lt_list = []

    # 开始拼接简单大图
    for i in range(len(data)):
        # 计算小图中心点相对像素坐标
        # x_m, y_m = gps2xy(la_list[i], lg_list[i], lb_la, lb_lg)
        # x = int(x_m / m_per_pixel)
        # x = concat_img_array.shape[0] - x 
        # y = int(y_m / m_per_pixel)

        x = int(((la_list[i] - lb_la) * meters_per_du_list[0]) / m_per_pixel)
        x = concat_img_array.shape[0] - x # 由于ndarray的x轴从上到下为正，这里需要取反调整
        y = int(((lg_list[i] - lb_lg) * meters_per_du_list[1]) / m_per_pixel)

        # 记录每张小图在画布上的中心像素坐标
        coordinates_list.append([x, y])

        img_path = os.path.join(dir_path, imgs_list[i])
        img_array = np.array(Image.open(img_path))

        # 旋转小图至地理正确
        if yaw_list[i] < 0:
            yaw = yaw_list[i] - 0.8
        elif yaw_list[i] > 0:
            yaw = yaw_list[i] - 1.0
        else:
            yaw = yaw_list[i]
        
        img_array = rotate_image(img_array, yaw)
        h, w, _ = img_array.shape

        # 定位至小图的左上角
        left_top_x, left_top_y = find_left_top(img_array, x, y)
        coordinates_lt_list.append([left_top_x, left_top_y])

        # 不重叠嵌入小图(根据左上角定位)
        concat_img_array[left_top_x:h + left_top_x, 
                         left_top_y:w + left_top_y] = np.where(
            (cv2.cvtColor(concat_img_array[left_top_x:h + left_top_x, 
                                           left_top_y:w + left_top_y], 
                                           cv2.COLOR_RGB2GRAY) == 0)[:, :, None],
            img_array,
            concat_img_array[left_top_x:h + left_top_x, 
                             left_top_y:w + left_top_y])
        
    # 保存, 拼接后大图保存为"数据文件夹名_concated_img.xxx"
    named_save_path = join(save_path ,f"{dir_name}_concated_img.jpeg")

    # 构造大图五点GPS的EXIF数据
    exif_dict = {"0th":{}, "Exif":{}, "GPS":{}, "1st":{}, "thumbnail":None}  # 初始化空EXIF字典
    avg_height_frac = Fraction(avg_height).limit_denominator(1000000)
    gps_exif = {
        piexif.GPSIFD.GPSLatitudeRef: b"N",
        piexif.GPSIFD.GPSLatitude: decimal_to_dms(center_la),
        piexif.GPSIFD.GPSLongitudeRef: b"E",
        piexif.GPSIFD.GPSLongitude: decimal_to_dms(center_lg),
        piexif.GPSIFD.GPSAltitudeRef: b"Above Sea Level",
        piexif.GPSIFD.GPSAltitude: (avg_height_frac.numerator, avg_height_frac.denominator)
    }
    exif_dict["GPS"] = gps_exif  # 写入EXIF数据到主IFD (0th)

    # 转换EXIF数据为二进制格式
    exif_bytes = piexif.dump(exif_dict)

    # 流式保存输出文件
    # pyvips.Image.new_from_array(concat_img_array).write_to_file(named_save_path)

    # Pillow保存文件
    Image.fromarray(concat_img_array).save(named_save_path)

    # 释放变量，清理内存
    del concat_img_array, img_array
    gc.collect() 

    # 注入EXIF数据（不可用于tif格式数据）
    piexif.insert(exif_bytes, named_save_path)

    return five_points_list



if __name__ == '__main__':
    s = time.time()

    # 航线数据文件夹路径和保存文件夹路径
    data_dir_path = "./data/img"
    save_dir_path = "./test"
    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)

    # 五点列表顺序依次为“中心点; 左下; 左上; 右下; 右上”
    points_list = main(data_dir_path, save_dir_path)

    e = time.time()
    print(f"Using time: {e - s:2f}")
    print(points_list)