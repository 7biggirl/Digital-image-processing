import boxFilter as mean
import numpy as np


def guideFilter(guideImg, srcImg, radius, eps):
    # in: 引导图的某一RGB平面guideImg，源图像的某一RGB平面srcImg，窗口大小radius，惩罚eps
    # out: 引导滤波后的某一平面dstImg

    h, w = srcImg.shape
    N = mean.boxFilter(np.ones((h, w)), radius)

    mean_I = mean.boxFilter(guideImg, radius) / N
    mean_p = mean.boxFilter(srcImg, radius) / N
    corr_I = mean.boxFilter(guideImg*guideImg, radius) / N
    corr_Ip = mean.boxFilter(guideImg*srcImg, radius) / N
    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = mean.boxFilter(a, radius) / N
    mean_b = mean.boxFilter(b, radius) / N

    dstImg = mean_a * guideImg + mean_b

    return dstImg

