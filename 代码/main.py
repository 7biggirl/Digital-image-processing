from guideFilter import guideFilter
from fastGuideFilter import fastGuideFilter
import cv2
import numpy as np
import math
import time


def RMSE(prediction, target):
    error = (prediction - target)**2  # 先求误差，**2 表示平方
    mse = np.mean(error)   # 求均方误差,np.mean()求均值
    rmse = np.sqrt(mse)   # 均方根误差,np.sqrt()表示开方
    return rmse


def PSNR(img1, img2):
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


if __name__ == "__main__":
    srcImg = cv2.imread("../peacock_window/rgb_doll_noise.png")
    guideImg = cv2.imread("../peacock_window/nir_flash.png")
    dstImg = cv2.imread("../peacock_window/GroundTruth_doll.png")

    bilImg = cv2.ximgproc.jointBilateralFilter(guideImg, srcImg, 20, 0.2, 8, borderType=cv2.BORDER_DEFAULT)  # doll
    # bilImg = cv2.ximgproc.jointBilateralFilter(guideImg, srcImg, 30, 2, 8, borderType=cv2.BORDER_DEFAULT)  # cave
    # bilImg = cv2.ximgproc.jointBilateralFilter(guideImg, srcImg, 8, 0.2, 8, borderType=cv2.BORDER_DEFAULT)  # teapot
    # bilImg = cv2.ximgproc.jointBilateralFilter(guideImg, srcImg, 10, 2, 8, borderType=cv2.BORDER_DEFAULT)  # bowl
    # bilImg = cv2.ximgproc.jointBilateralFilter(guideImg, srcImg, 10, 0.2, 8, borderType=cv2.BORDER_DEFAULT)  # book

    cv2.imshow("bliateralImg", bilImg)
    cv2.imwrite("../peacock_window/bliateralImg.png", bilImg)

    guasImg = cv2.GaussianBlur(srcImg, (5, 5), 0)
    cv2.imshow("guasImg", guasImg)
    cv2.imwrite("../peacock_window/guasImg.png", guasImg)


    srcImg = srcImg.astype(np.float64) / 255
    guideImg = guideImg.astype(np.float64) / 255

    radius = 5
    eps = 0.000001
    outImg = np.zeros(srcImg.shape, dtype=float)

    start = time.time()
    outImg[:, :, 2] = guideFilter(guideImg[:, :, 2], srcImg[:, :, 2], radius, eps)
    outImg[:, :, 1] = guideFilter(guideImg[:, :, 1], srcImg[:, :, 1], radius, eps)
    outImg[:, :, 0] = guideFilter(guideImg[:, :, 0], srcImg[:, :, 0], radius, eps)
    print("guideFilter cost", time.time()-start, "seconds")

    size = 2
    fastOutImg = np.zeros(srcImg.shape, dtype=float)

    start = time.time()
    fastOutImg[:, :, 2] = fastGuideFilter(guideImg[:, :, 2], srcImg[:, :, 2], radius, eps, size)
    fastOutImg[:, :, 1] = fastGuideFilter(guideImg[:, :, 1], srcImg[:, :, 1], radius, eps, size)
    fastOutImg[:, :, 0] = fastGuideFilter(guideImg[:, :, 0], srcImg[:, :, 0], radius, eps, size)
    print("fastGuideFilter cost", time.time() - start, "seconds\n")

    print("RMSE(outImg, dstImg):", RMSE(outImg, dstImg))
    print("RMSE(fastOutImg, dstImg):", RMSE(fastOutImg, dstImg))
    print("RMSE(fastOutImg, outImg):", RMSE(fastOutImg, outImg))
    print("RMSE(guasImg, dstImg):", RMSE(guasImg, dstImg))
    print("RMSE(bilImg, dstImg):", RMSE(bilImg, dstImg), "\n")

    print("PSNR(srcImg, dstImg):", PSNR(srcImg, dstImg))
    print("PSNR(outImg ,dstImg):", PSNR(outImg, dstImg))
    print("PSNR(fastOutImg, dstImg):", PSNR(fastOutImg, dstImg))
    print("PSNR(guasImg, dstImg):", PSNR(guasImg, dstImg))
    print("PSNR(bilImg, dstImg):", PSNR(bilImg, dstImg), "\n")

    cv2.imshow("../peacock_window/outImg", outImg)
    cv2.imshow("../peacock_window/fastOutImg", fastOutImg)
    cv2.imwrite("../peacock_window/outImg.png", np.uint8(outImg * 255))
    cv2.imwrite("../peacock_window/fastOutImg.png", np.uint8(fastOutImg * 255))
    cv2.waitKey()

