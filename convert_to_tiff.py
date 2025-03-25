import os

import piexif
import numpy as np
from osgeo import gdal, osr
from PIL import Image, ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


def extract_gps_coordinates(jpg_path):
    """
    从JPG中提取GPS坐标并转换为标准格式
    """
    try:
        # 读取EXIF数据
        exif_dict = piexif.load(Image.open(jpg_path).info['exif'])

        if "GPS" not in exif_dict:
            print(f"未找到GPS信息: {jpg_path}")
            return None

        gps = exif_dict["GPS"]

        # 提取经纬度
        if all(id in gps for id in [2, 4]):  # 确保有经纬度数据
            # 解析纬度
            lat = gps[2]
            lat_ref = gps[1].decode('ascii')
            lat_deg = float(lat[0][0]) / float(lat[0][1])
            lat_min = float(lat[1][0]) / float(lat[1][1])
            lat_sec = float(lat[2][0]) / float(lat[2][1])
            latitude = lat_deg + lat_min / 60.0 + lat_sec / 3600.0
            if lat_ref == 'S':
                latitude = -latitude

            # 解析经度
            lon = gps[4]
            lon_ref = gps[3].decode('ascii')
            lon_deg = float(lon[0][0]) / float(lon[0][1])
            lon_min = float(lon[1][0]) / float(lon[1][1])
            lon_sec = float(lon[2][0]) / float(lon[2][1])
            longitude = lon_deg + lon_min / 60.0 + lon_sec / 3600.0
            if lon_ref == 'W':
                longitude = -longitude

            # 如果有高度信息也提取
            altitude = None
            if 6 in gps:
                alt = gps[6]
                altitude = float(alt[0]) / float(alt[1])

            return {"latitude": latitude, "longitude": longitude, "altitude": altitude}
    except Exception as e:
        print(f"提取GPS信息时出错: {str(e)}")
        return None


def create_geotiff(jpg_path, output_path, gps_info):
    """
    创建正确的GeoTIFF文件
    """
    try:
        # 打开源图像
        ds = gdal.Open(jpg_path)
        if ds is None:
            print(f"无法打开源文件: {jpg_path}")
            return False

        # 获取图像尺寸
        width = ds.RasterXSize
        height = ds.RasterYSize

        # 创建输出
        driver = gdal.GetDriverByName('GTiff')
        dst_ds = driver.Create(output_path,
                               width,
                               height,
                               ds.RasterCount,
                               gdal.GDT_Byte,
                               options=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=YES'])

        # 设置投影
        srs = osr.SpatialReference()
        srs.SetWellKnownGeogCS('WGS84')
        dst_ds.SetProjection(srs.ExportToWkt())

        # 设置地理变换
        # 这里需要根据实际情况计算像素分辨率
        # 这只是一个示例，实际应用中需要更精确的计算
        pixel_size_x = 0.0001  # 大约11米/像素
        pixel_size_y = -0.0001  # 负值因为y轴方向相反

        geotransform = [
            gps_info['longitude'],  # 左上角x坐标
            pixel_size_x,  # 水平像素分辨率
            0,  # 旋转角度, 0表示图像"向上"
            gps_info['latitude'],  # 左上角y坐标
            0,  # 旋转角度, 0表示图像"向上"
            pixel_size_y  # 垂直像素分辨率
        ]
        dst_ds.SetGeoTransform(geotransform)

        # 复制图像数据
        for i in range(ds.RasterCount):
            band = ds.GetRasterBand(i + 1)
            data = band.ReadAsArray()
            dst_band = dst_ds.GetRasterBand(i + 1)
            dst_band.WriteArray(data)

        # 确保写入磁盘
        dst_ds.FlushCache()
        print(f"成功创建GeoTIFF: {output_path}")
        return True

    except Exception as e:
        print(f"创建GeoTIFF时出错: {str(e)}")
        return False
    finally:
        # 清理资源
        dst_ds = None
        ds = None


def convert_jpg_to_geotiff(jpg_path):
    """
    将JPG转换为正确的GeoTIFF
    """
    try:
        # 首先提取GPS信息
        gps_info = extract_gps_coordinates(jpg_path)
        if not gps_info:
            print(f"无法提取GPS信息，跳过转换: {jpg_path}")
            return

        # 设置输出路径
        output_path = os.path.splitext(jpg_path)[0] + '.tiff'

        # 创建GeoTIFF
        if create_geotiff(jpg_path, output_path, gps_info):
            print(f"成功完成转换: {jpg_path} -> {output_path}")
            print(f"GPS信息: 纬度={gps_info['latitude']}, 经度={gps_info['longitude']}")
            if gps_info['altitude']:
                print(f"海拔: {gps_info['altitude']}米")

    except Exception as e:
        print(f"转换失败: {jpg_path}")
        print(f"错误信息: {str(e)}")


def convert_folder(folder_path):
    """
    转换文件夹中的所有JPG图像
    """
    try:
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.jpg', '.jpeg')):
                jpg_path = os.path.join(folder_path, filename)
                convert_jpg_to_geotiff(jpg_path)
    except Exception as e:
        print(f"处理文件夹时出错: {str(e)}")


if __name__ == "__main__":
    # 替换为您的文件夹路径
    folder_path = "./test"
    convert_folder(folder_path)