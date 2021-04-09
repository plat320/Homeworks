import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image



def show(img, fig, title):
    # plt.imshow(Image.fromarray(np.uint8(a), 'RGB'))
    # Image.fromarray(np.uint8(a), 'RGB').show(name)
    plt.imsave(title +".png", np.uint8(img))
    fig.imshow(Image.fromarray(np.uint8(img), 'RGB'))
    fig.set_title(title)
    fig.axis("off")


def Retinex(img, sig):
    return np.log10(img) - np.log10(cv2.GaussianBlur(img, (0, 0), sig))     # 전체에 대한 가우시

def Convert255(img, mode):

    if mode == "org":
        return (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
    elif mode =="CR":
        G = 192
        b = 30
        return G * (img/255+b)


if __name__ == '__main__':
    img_dir = "./example"
    for i in range(1,3):
        img_name_L = "{}_L.ppm".format(i)
        img_name_R = "{}_R.ppm".format(i)
        # orgimg = Image.open(os.path.join(img_dir, img_name))         # w, h, c
        orgimg_L = cv2.imread(os.path.join(img_dir, img_name_L), cv2.IMREAD_GRAYSCALE)  # CV_8UC1
        orgimg_R = cv2.imread(os.path.join(img_dir, img_name_R), cv2.IMREAD_GRAYSCALE)

        BM = cv2.StereoBM_create(numDisparities=64, blockSize=25)       # numDisparities -> 16배 수, blocksize -> odd
        SGBM = cv2.StereoSGBM_create(numDisparities=64, blockSize=25)

        disparity_BM = BM.compute(orgimg_L, orgimg_R)
        disparity_SGBM = SGBM.compute(orgimg_L, orgimg_R)

        plt.figure(figsize=(10, 6))
        plt.subplot(221)
        plt.imshow(orgimg_L, cmap="gray")
        plt.title("org L")

        plt.subplot(222)
        plt.imshow(orgimg_R, cmap="gray")
        plt.title("org R")

        plt.subplot(223)
        plt.imshow(disparity_BM, cmap="gray")
        plt.title("disparity BM")

        plt.subplot(224)
        plt.imshow(disparity_SGBM, cmap="gray")
        plt.title("disparity SGBM")

        plt.show()
        # plt.savefig("{}.png".format(i))
