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


if __name__ == '__main__':
    img_dir = "./example"
    img_name = "11.jpg"
    general_gamma = 1.256
    # a = cv2.imread(os.path.join(img_dir, img_name))
    # cv2.imwrite(os.path.join(img_dir, "1.jpg"), a)

    fig = plt.figure()
    rows = 2
    cols = 2

    orgimg = cv2.imread(os.path.join(img_dir, img_name))
    orgimg = cv2.cvtColor(orgimg, cv2.COLOR_BGR2RGB)

    img = np.float64(orgimg)/255.0
    cor_img = img**general_gamma
    show(orgimg, fig.add_subplot(rows, cols, 1), "orginal image")
    # show(np.uint8(255 * cor_img), fig.add_subplot(rows, cols, 2), "gamma image")        # general gamma correction

    tau = 2


    HSV_img = cv2.cvtColor(orgimg, cv2.COLOR_RGB2HSV)


    hist = cv2.calcHist([HSV_img[:,:,2]], [0],None,[256],[0,256])
    # hist = cv2.calcHist([orgimg], [0],None,[256],[0,256])
    fig.add_subplot(rows, cols, 3).plot(hist[:,0])
    hist = hist/np.sum(hist)
    plt.plot(hist)
    mean = np.mean(hist)
    num = 0
    for n,i in enumerate(hist):
        num+=n*i
    mean = num
    num = 0
    for n, i in enumerate(hist):
        num += (n-mean)**2 * i
    sigma = np.sqrt(num)/256
    mean = mean/256
    Flag_low = True if sigma *4 <1/tau else False
    Flag_bright = True if mean >= 0.5 else False

    if Flag_low == True:
        gamma = -np.log2(sigma)
        if Flag_bright == True:
            HSV_img[:,:,2] = (HSV_img[:,:,2]/255) ** gamma *255
        else:
            # cor_value = np.zeros_like(HSV_img[:,:,2])
            for x in range(HSV_img.shape[0]):
                for y in range(HSV_img.shape[1]):
                    HSV_img[x,y,2] = 255*((HSV_img[x,y,2]/255)**(1/gamma)) / ((HSV_img[x,y,2]/255)**(1/gamma) + (1-(HSV_img[x,y,2]/255)**(1/gamma)) * mean**(1/gamma))
    else:
        gamma = np.exp((1-(mean+sigma))/2)
        HSV_img[:,:,2] = ((HSV_img[:,:,2]/255) ** (1/gamma)) *255
    hist = cv2.calcHist([np.uint8(HSV_img[:,:,2])], [0],None,[256],[0,256])
    fig.add_subplot(rows, cols, 4).plot(hist)

    print("mean = {}, sigma = {}, gamma = {}".format(mean, sigma, gamma))
    correction_img = cv2.cvtColor(HSV_img, cv2.COLOR_HSV2RGB)
    show(np.uint8(correction_img), fig.add_subplot(rows, cols, 2), "correction image")
    # plt.show()
    plt.savefig("./123.png")