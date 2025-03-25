import cv2
import numpy as np
from PIL import Image


def cropImgsPair(x, y, img_array, simple_concated_img) -> np.ndarray:
    '''
        传入小图中心坐标返回其对应的周围大图的array, 
        并对大图做出部分mask,只保留小图边缘的像素参与后续特征点计算
    '''
    crop_scale = 1.2

    h, w, _ = img_array.shape

    img_x_lt = int(x - h * 0.5)
    img_y_lt = int(y - w * 0.5)

    large_img_x_lt = int(x - h * 0.5 * crop_scale)
    large_img_y_lt = int(y - w * 0.5 * crop_scale)

    img_x_lt_rel = img_x_lt - large_img_x_lt
    img_y_lt_rel = img_y_lt - large_img_y_lt

    large_img = simple_concated_img[large_img_x_lt:large_img_x_lt + int(crop_scale * h), 
                                    large_img_y_lt:large_img_y_lt + int(crop_scale * w), 
                                    :]
    
    # 构建大图的mask, 只计算大图中不在小图部分的外圈的特征点
    reverse_masked_img = computeReverseMask(img_array)
    mask = np.zeros((large_img.shape[0], large_img.shape[1]), dtype=np.uint8)
    mask[:, :] = 255
    mask[img_x_lt_rel:img_x_lt_rel + h, 
         img_y_lt_rel:img_y_lt_rel + w] = reverse_masked_img

    # bug_test!
    t = Image.fromarray(large_img)
    mask = Image.fromarray(mask)
    t.save('./test.jpeg')
    mask.save('./mask_test.jpeg')
    breakpoint()

    return large_img, mask, large_img_x_lt, large_img_y_lt

def computeReverseMask(img_array):
    # 返回相同大小但通道为1的ndarray, 值为255的部分代表没有图像, 0代表有图像
    h, w, _ = img_array.shape
    mask = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY) != 0 

    masked_img = np.zeros((h, w), dtype=np.uint8)
    masked_img[:, :] = 255
    masked_img[mask] = 0

    return masked_img

# 检测A、B图片的SIFT关键特征点，并计算特征描述子
def detectAndDescribe(image, mask=None):
    # 建立SIFT生成器
    sift = cv2.SIFT_create(nfeatures=0,
                           nOctaveLayers=4, 
                           contrastThreshold=0.03, 
                           edgeThreshold=12, 
                           sigma=1.6)
    # 检测SIFT特征点，并计算描述子
    (kps, features) = sift.detectAndCompute(image, mask)
    # 将结果转换成NumPy数组
    kps = np.float32([kp.pt for kp in kps])
    # 返回特征点集，及对应的描述特征
    return (kps, features)

def matchLargeAndSmall(large_img, unaligned_img, large_img_mask=None, small_img_mask=None):
    # 对传入的ndarray转换为BGR格式
    large_img = cv2.cvtColor(large_img, cv2.COLOR_RGB2BGR)
    unaligned_img = cv2.cvtColor(unaligned_img, cv2.COLOR_RGB2BGR)

    kpsA, featuresA = detectAndDescribe(large_img)

    # 只计算小图中有图像部分的SIFT特征点
    # unaligned_img_mask_bool = cv2.cvtColor(unaligned_img, 
    #                                        cv2.COLOR_BGR2GRAY).astype(bool)
    # unaligned_img_mask = np.zeros((unaligned_img.shape[0], unaligned_img.shape[1]), 
    #                               dtype=np.uint8)
    # unaligned_img_mask[unaligned_img_mask_bool] = 255
    kpsB, featuresB = detectAndDescribe(unaligned_img, small_img_mask)

    # 建立快速最近邻匹配器
    FLANN_INDEX_KDTREE = 1 # kd树
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=7)
    search_params = dict(checks=70)
    bf = cv2.FlannBasedMatcher(index_params, search_params)

    # 使用KNN检测来自A、B图的SIFT特征匹配对，K=2
    print(len(featuresA), len(featuresB))
    matches = bf.knnMatch(featuresB, featuresA, 2)

    good = []
    for m in matches:
        # 当最近距离跟次近距离的比值小于ratio值时，保留此匹配对
        if len(m) == 2 and m[0].distance < m[1].distance * 0.8:
            # 存储两个点在featuresA, featuresB中的索引值
            good.append((m[0].trainIdx, m[0].queryIdx))
        # good.append((m.trainIdx, m.queryIdx))
    print(f"\ngood matches: {len(good)}")

    # 获取匹配对的点坐标
    ptsA = np.float32([kpsA[i] for (i, _) in good])
    ptsB = np.float32([kpsB[i] for (_, i) in good])

    # 计算未对齐小图到大图的视角变换矩阵
    M, _ = cv2.findHomography(ptsB, ptsA, cv2.RANSAC, 1.0)
    print(M.shape, M.dtype)

    h, w, _ = large_img.shape

    aligned_img = cv2.warpPerspective(unaligned_img, 
                                      M, 
                                      (w, h))

    # 计算aligned_img的掩码并抠掉其在large_img上的位置
    # aligned_img中非黑色位置为True
    aligned_img_mask = aligned_img.astype(bool)
    large_img[aligned_img_mask] = 0

    # 贴合aligned的图像和masked后的large图像（相当于逐元素相加）
    result = cv2.addWeighted(large_img, 1.0, aligned_img, 1.0, 0)

    # cv2.imwrite("./unaligned_img.jpg", unaligned_img)
    # cv2.imwrite("./aligned_img.jpg", aligned_img)

    # cv2.imwrite("./test_result.jpg", result)
    # 对传出的ndarray转换为RGB格式
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    return result



if __name__ == '__main__':
    large_img = cv2.imread("./result.jpg")
    unaligned_img = cv2.imread("./img/DJI_20240807093411_0002_V.jpeg")

    matchLargeAndSmall(large_img, unaligned_img)

