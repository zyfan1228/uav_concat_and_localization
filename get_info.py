import os

import pandas as pd
from pyexiv2 import Image


def turn2float(strList: list) -> list:
    # 将分子分母化的经纬度信息转换为浮点数表示
    gps_list = []
    strlist = strList.split()

    for i in range(3):
        data_list = strlist[i].split("/")
        gps_list.append([float(data_list[0]), float(data_list[1])])

    du = gps_list[0][0] / gps_list[0][1]
    fen = gps_list[1][0] / gps_list[1][1]
    miao = gps_list[2][0] / gps_list[2][1]

    # 转换为十进制表示
    decimal_data = decimal(du, fen, miao)

    return [abs(du), fen, miao], decimal_data


def decimal(du, fen, miao) -> float:
    # 度分秒转换为十进制
    return abs(du) + fen / 60 + miao / 3600


def get_info(imgs_path):
    image = Image(imgs_path)
    xmp = image.read_xmp()
    exif = image.read_exif()

    # 纬度
    latitude_info = exif["Exif.GPSInfo.GPSLatitude"]
    # 经度
    longitude_info = exif["Exif.GPSInfo.GPSLongitude"]

    # # 纬度标志
    # latitude_ref_info = exif["Exif.GPSInfo.GPSLatitudeRef"]
    # # 经度标志
    # longitude_ref_info = exif["Exif.GPSInfo.GPSLongitudeRef"]

    latitude, la_decimal = turn2float(latitude_info)
    longitude, lg_decimal = turn2float(longitude_info)

    list.extend(latitude, longitude)
    info_list = latitude
    info_list.append(la_decimal)
    info_list.append(lg_decimal)

    # 水平转角yaw: 地理南向为起点0，正西方向为+90，正东方向为-90，定义(-180,180)
    yaw = float(xmp['Xmp.drone-dji.GimbalYawDegree'])
    info_list.append(yaw)

    # width = exif["EXIF ExifImageWidth"].values[0]
    # height = exif["EXIF ExifImageLength"].values[0]

    # 获取无人机的焦距
    length_frac = str(exif["Exif.Photo.FocalLength"]).split("/")
    length = float(length_frac[0]) / float(length_frac[1])
    
    info_list.append(length)

    relative_altitude = float(xmp["Xmp.drone-dji.RelativeAltitude"])  
    # 相对地秒高度，单位：m
    info_list.append(relative_altitude)

    # # 中点坐标
    # x, y = width // 2, height // 2

    image.close()

    return info_list


def save_csv(imgs_path, save_path):
    dir_path = imgs_path
    dir_name = os.path.split(dir_path)[1]
    imgs_list = os.listdir(dir_path)

    info_list = []
    for i in range(len(imgs_list)):
        img_path = os.path.join(dir_path, imgs_list[i])
        info = get_info(img_path)
        info_list.append(info)

    name_attr = ['la_du', 'la_fen', 'la_miao', 'lg_du', 'lg_fen', 'lg_miao',
                 'la_decimal', 'lg_decimal', 'yaw', 'focalLength', 'rel_altitude']
    csv_writer = pd.DataFrame(columns=name_attr, data=info_list)
    named_save_path = os.path.join(save_path, f'{dir_name}_info.csv')
    csv_writer.to_csv(named_save_path, encoding='utf-8', index=False)



if __name__ == "__main__":
    pass
