import cv2
import numpy as np


def boxFilter(srcImg, radius):
    # in: 源图像的某一平面srcImg， 窗口大小radius
    # out： 方框滤波的图像某一平面dstImg
    dstImg = np.zeros(srcImg.shape)
    height, width = srcImg.shape[0], srcImg.shape[1]

    # 列累加
    sumY = np.cumsum(srcImg, axis=0)
    dstImg[0:radius + 1, :] = sumY[radius:2 * radius + 1, :]
    dstImg[radius + 1:height - radius, :] = sumY[2 * radius + 1:height, :] - sumY[0:height - 2 * radius - 1, :]
    dstImg[height - radius:height, :] = np.tile(sumY[height - 1, :], (radius, 1)) - sumY[height - 2 * radius - 1:height - radius - 1, :]

    # 按行累加
    sumX = np.cumsum(dstImg, axis=1)
    dstImg[:, 0:radius + 1] = sumX[:, radius:2 * radius + 1]
    dstImg[:, radius + 1:width - radius] = sumX[:, 2 * radius + 1:width] - sumX[:, 0:width - 2 * radius - 1]
    dstImg[:, width - radius:width] = np.tile(sumX[:, width - 1][:, np.newaxis], (1, radius)) - sumX[:, width - 2 * radius - 1:width - radius - 1]

    return dstImg


