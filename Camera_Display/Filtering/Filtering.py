import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from skimage.util import random_noise
from PIL import Image



def show_gray(img, fig, title):
    # plt.imshow(Image.fromarray(np.uint8(a), 'RGB'))
    # Image.fromarray(np.uint8(a), 'RGB').show(name)
    plt.imsave(title +".png", np.uint8(img), cmap="gray")
    fig.imshow(Image.fromarray(np.uint8(img), "L"), cmap = "gray")
    fig.set_title(title)
    fig.axis("off")

def show_RGB(img, fig, title):
    # plt.imshow(Image.fromarray(np.uint8(a), 'RGB'))
    # Image.fromarray(np.uint8(a), 'RGB').show(name)
    plt.imsave(title +".png", np.uint8(img))
    fig.imshow(Image.fromarray(np.uint8(img)))
    fig.set_title(title)
    fig.axis("off")

def sp_noise(image,prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output



if __name__ == '__main__':
    # img_dir = "./example"
    # img_name = "13.jpg"
    #
    # general_gamma = 1.256
    # a = cv2.imread(os.path.join(img_dir, img_name))
    # cv2.imwrite(os.path.join(img_dir, "1.jpg"), a)

    fig = plt.figure()
    rows = 2
    cols = 2



    img_dir = "./example"
    img_name = "BF.jpg"        # 1은 png 2는 bmp

    img = cv2.imread(os.path.join(img_dir, img_name),cv2.IMREAD_COLOR)
    img_tmp = img.copy()
    noise_img = sp_noise(img, 0.1)
    # noise_img = np.uint8(random_noise(img, mode="gaussian") * 255)
    show_RGB(img, fig.add_subplot(rows,cols,1), "original image")
    # show_gray(noise_img, fig.add_subplot(rows,cols,2), "noisy image")
    gaussian = cv2.GaussianBlur(noise_img, (9,9), 2)
    BF = cv2.bilateralFilter(noise_img, 9, 75, 75)
    show_RGB(gaussian, fig.add_subplot(rows, cols, 2), "gaussian")

    MF = cv2.medianBlur(noise_img, 3)
    kernel = np.ones((5,5), np.float32)/25
    AF = cv2.filter2D(noise_img, -1, kernel)


    show_gray(MF, fig.add_subplot(rows,cols,3), "MF")
    show_gray(AF, fig.add_subplot(rows,cols,4), "AF")
    # show_RGB(BF, fig.add_subplot(rows,cols,3),"BF")
    # cv2.imshow("asd",dct)
    # cv2.waitKey()
    plt.show()