# from PIL import Image
# from math import cos, sin
import math
import imutils

from get_info import get_info


def decimal_to_dms(decimal):
    # 转换浮点经纬度为度分秒格式
    degree = int(decimal)
    minute = int((decimal - degree) * 60)
    second = ((decimal - degree) * 60 - minute) * 60

    return f"{degree}°{minute}'{second}''"

def rotate_image(image, image_yaw):
    # 传入img和yaw角度，返回地理正确旋转之后的图片(上北下南)
    rotate_angle = image_yaw + 180
    image_rotated = imutils.rotate_bound(image, rotate_angle)

    return image_rotated

def compute_m_per_pixel(sigma, f_length, height) -> float:
    '''
        args:
            f_length: 焦距(mm);
            height: 无人机飞行相对地面高度(m) (非海拔高度);
        return:
            返回每个像素代表的实际长度(m)
    '''
    return sigma * height / f_length * 1e-3

def located(pixel_x, pixel_y, 
            ref_lat, ref_lon, 
            f_length, height, 
            p, 
            drone='M3TD'):
    
    '''
    x, y 分别为行列坐标
    '''

    a = 6378137.0  # 赤道半径 (米)
    b = 6356752.3142  # 极半径 (米)
    e = math.sqrt(1 - (b / a) ** 2)  # 偏心率

    h, w = 3024, 4032

    if drone == 'M3TD':
        # M3TD 像元大小约为1.27μm，但这里是用的 1/2 最大分辨率去拍摄图片，所以等价于1.27 * 2μm
        sigma = 2.55
    elif drone == 'MT03':
        pass
   
    m_per_pixel_x = compute_m_per_pixel(sigma, f_length, height)
    m_per_pixel_y = m_per_pixel_x  * 0.95

    phi = math.radians(ref_lat)
    # 计算ref_lat附近的1度纬度的长度
    meters_per_degree_lat = (math.pi / 180) * a * (1 - e**2) / \
        (1 - e**2 * math.sin(phi)**2)**(3/2)
    # 计算ref_lat附近的1度经度的长度
    meters_per_degree_lon = (math.pi / 180) * a * math.cos(phi) / \
        math.sqrt(1 - e**2 * math.sin(phi)**2) 

    # d_x, d_y 为给定坐标点相对中心点的坐标增量
    mid_x, mid_y = h // 2, w // 2
    dx, dy = mid_x - pixel_x, pixel_y - mid_y

    # 计算按地理方向旋转后的新像素坐标。此时符合上北下南
    rotated_dx = math.cos(p) * dx + math.sin(p) * dy
    rotated_dy = -math.sin(p) * dx + math.cos(p) * dy
    # breakpoint()

    dx_m, dy_m = rotated_dx * m_per_pixel_x, rotated_dy * m_per_pixel_y

    lat_out = ref_lat + dx_m / meters_per_degree_lat
    lon_out = ref_lon + dy_m / meters_per_degree_lon

    return lat_out, lon_out

def main(pixel_x, pixel_y, img_path):
    '''
    传入的坐标非上北下南，而是未旋转图片上选取的坐标
    '''
    info_list = get_info(img_path)

    ref_lat, ref_lon = info_list[6], info_list[7]
    yaw = int(info_list[8]) 
    f_length = info_list[9]
    height = info_list[10]

    # img_array = np.array(Image.open(img_path))
    # rotated_img_array = rotate_image(img_array, yaw)

    yaw = abs(yaw) - 180 if yaw < 0 else 180 - yaw

    lat_out, lon_out = located(pixel_x, pixel_y, 
                               ref_lat, ref_lon, 
                               f_length, height, 
                               p=math.radians(yaw), drone='M3TD')
    
    # 若要输出度分秒格式，请启用下面两行
    # lat_out = decimal_to_dms(lat_out)
    # lon_out = decimal_to_dms(lon_out)
    
    print(f"像素坐标 ({pixel_y}, {pixel_x}) 对应的经纬度为：({lat_out},{lon_out})")
    # print(f"像素坐标 ({pixel_x}, {pixel_y}) 对应的经纬度为：({lon_out},{lat_out})")



if __name__ == '__main__':
    img_path = '.\data\jingweidu\DJI_20241201140559_0001_V.jpeg'
    # 以下坐标为原图（即未旋转的原始航线数据）上选取的坐标信息。
    # 左上角为(0, 0)点，X轴从左至右为正，Y轴从上至下为正
    pixel_point = (3380,1513)
    
    # 这里输入顺序交换了一下，以满足实际需求
    pixel_x = pixel_point[1]
    pixel_y = pixel_point[0]

    main(pixel_x, pixel_y, img_path)
