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
    img_name = "3.jpg"
    orgimg = Image.open(os.path.join(img_dir, img_name))         # w, h, c

    beta = 46
    alpha = 125

    fig = plt.figure()
    rows = 2
    cols = 2

    show(orgimg, fig.add_subplot(rows, cols, 1), "orginal image")

    #### img 화소값 더하고 채널로나눔
    orgimg = np.float64(np.asarray(orgimg)) +1                      # convert to float64 +1는 log0일 경우방지
    img = np.sum(orgimg, axis = 2) / orgimg.shape[2]

    #### get retinex
    SSR = Retinex(img, 80)
    MSR = (Retinex(img, 15) + Retinex(img, 80) + Retinex(img, 250))/3
    append_img = np.zeros((img.shape[0], img.shape[1], 3))
    for i in range(3):              # dimension 맞아야 연산됨
        append_img[:,:,i] = img
    MSRCR = beta * (np.log10(alpha * orgimg) - np.log10(append_img))        # color restoration

    img = np.expand_dims(img, 2)
    SSR = np.expand_dims(SSR, 2)
    MSR = np.expand_dims(MSR, 2)

    #### display하기위해 convert
    SSR = Convert255(SSR, "org")
    MSR = Convert255(MSR, "org")
    MSRCR = Convert255(MSRCR, "org")

    SSR_display = np.zeros_like(orgimg)
    MSR_display = np.zeros_like(orgimg)
    MSRCR_display = np.zeros_like(orgimg)

    for x in range(SSR.shape[0]):
        for y in range(SSR.shape[1]):
            SSR_mul = SSR[x,y,0] / img[x,y,0]
            MSR_mul = MSR[x,y,0] / img[x,y,0]
            MSRCR_mul = MSR[x,y,0] / img[x,y,0]
            for i in range(orgimg.shape[2]):        # 255 못넘게 upper bound 제한
                SSR_display[x,y,i] = 255 if SSR_mul * orgimg[x,y,i] > 255 else SSR_mul * orgimg[x,y,i]
                MSR_display[x,y,i] = 255 if MSR_mul * orgimg[x,y,i] > 255 else MSR_mul * orgimg[x,y,i]
                MSRCR_display[x,y,i] = 255 if MSRCR_mul * orgimg[x,y,i] > 255 else MSRCR_mul * orgimg[x,y,i]

    MSRCR = Convert255(MSRCR_display/255, "CR")     # cannonical gain/offset

    show(SSR_display, fig.add_subplot(rows, cols, 2), "SSR")
    show(MSR_display, fig.add_subplot(rows, cols, 3), "MSR")
    show(MSRCR_display, fig.add_subplot(rows, cols, 4), "MSRCR")

    plt.show()
    plt.savefig("123.png")
