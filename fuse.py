import os

import numpy as np
import cv2
import pandas as pd
import tqdm
from PIL import Image

from fuse_utils import *


if __name__ == '__main__':


    # # 开始进行融合
    # for i in tqdm.tqdm(range(len(data)), ncols=100, colour="green"):
    #     # # test
    #     # if i > 2:
    #     #     break

    #     img_path = os.path.join(dir_path, imgs_list[i])
    #     img_array = np.array(Image.open(img_path))
    #     # 旋转小图片至地理正确
    #     img_array = rotate_image(img_array, yaw_list[i])

    #     # 小图片中心坐标
    #     [x, y] = coordinates_list[i]
    #     # 裁剪出的包含小图的大图
    #     large_img_array, large_mask, lxlt, lylt = cropImgsPair(x, y, 
    #                                                      img_array, 
    #                                                      concat_img_array)
    #     h, w, _ = large_img_array.shape
        
    #     # 小图和大图进行融合
    #     fused_img_part = matchLargeAndSmall(large_img_array, img_array, large_mask)

    #     # 融合后图片拼回全图中
    #     concat_img_array[lxlt:lxlt+h, lylt:lylt+w] = fused_img_part

    # concated_img_new = Image.fromarray(concat_img_array)
    # concated_img_new.save(f"./{dir_name}_fused_img.jpeg")


    # concat_img = Image.fromarray(concat_img_array)
    # concat_img.save(f'./{dir_name}_concat.jpeg')