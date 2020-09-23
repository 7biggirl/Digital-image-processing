import numpy as np
import boxFilter as mean
import cv2


def fastGuideFilter(guideImg, srcImg, radius, eps, size):
    # in: 引导图的某一RGB平面guideImg，源图像的某一RGB平面srcImg，窗口大小radius，惩罚eps，变化倍数size
    # out: 引导滤波后的某一平面dstImg
    h, w = srcImg.shape
    sub_I = cv2.resize(guideImg, (int(w/size), int(h/size)))
    sub_p = cv2.resize(srcImg, (int(w/size), int(h/size)))
    sub_r = radius / size

    N = mean.boxFilter(np.ones(sub_I.shape), int(sub_r))

    mean_I = mean.boxFilter(sub_I, int(sub_r)) / N
    mean_p = mean.boxFilter(sub_p, int(sub_r)) / N
    corr_I = mean.boxFilter(sub_I*sub_I, int(sub_r)) / N
    corr_Ip = mean.boxFilter(sub_I*sub_p, int(sub_r)) / N

    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.resize(mean.boxFilter(a, int(sub_r)) / N, (w,h))
    mean_b = cv2.resize(mean.boxFilter(b, int(sub_r)) / N, (w,h))

    return mean_a * guideImg + mean_b
