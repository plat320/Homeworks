import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image



def show(img, fig, title):
    # plt.imshow(Image.fromarray(np.uint8(a), 'RGB'))
    # Image.fromarray(np.uint8(a), 'RGB').show(name)
    plt.imsave(title +".png", np.uint8(img), cmap='gray')
    fig.imshow(Image.fromarray(np.uint8(img), 'L'),cmap='gray')
    fig.set_title(title)
    fig.axis("off")


if __name__ == '__main__':
    img_dir = "./example"
    img_name = "13.jpg"

    general_gamma = 1.256
    # a = cv2.imread(os.path.join(img_dir, img_name))
    # cv2.imwrite(os.path.join(img_dir, "1.jpg"), a)

    fig = plt.figure()
    rows = 2
    cols = 2

    '''
    orgimg = cv2.imread(os.path.join(img_dir, img_name))
    orgimg = cv2.cvtColor(orgimg, cv2.COLOR_BGR2RGB)

    show(orgimg, fig.add_subplot(rows,cols,1), "original image")

    scale = 1.5
    x = int(orgimg.shape[0] * 1.5)
    y = int(orgimg.shape[1] * 1.5)


    near_result = np.zeros((x,y,orgimg.shape[2]))
    bil_result = np.zeros_like(near_result)
    bic_result = np.zeros_like(near_result)

    bic_mat = np.array([
        [-1,1,-1,1], [0,0,0,1], [1,1,1,1], [8,4,2,1]
    ])


    img = cv2.copyMakeBorder(orgimg, 1,1,1,1,cv2.BORDER_REFLECT)
    for x in range(near_result.shape[0]):
        for y in range(near_result.shape[1]):
            a = math.floor(x/scale)
            b = math.floor(y/scale)

            #### bilinear
            alpha = x/scale - a
            beta = y/scale - b

            ff = img[a+1,b+1,:]
            fc = img[a+1,b+2,:]
            cf = img[a+2,b+1,:]
            cc = img[a+2,b+2,:]
            near_result[x,y,:] = img[round(x/scale), round(y/scale),:]
            bil_result[x,y,:] = (alpha*beta*cc+(1-alpha)*beta*fc+alpha*(1-beta)*cf+(1-alpha)*(1-beta)*ff).astype(dtype=int)

    img = cv2.copyMakeBorder(img, 1,1,1,1,cv2.BORDER_REFLECT)
    for x in range(near_result.shape[0]):
        for y in range(near_result.shape[1]):
            for channel in range(img.shape[2]):
                x_a = math.floor(x / scale)
                y_a = math.floor(y / scale)
                #### bicubic
                xscale = x/scale-x_a
                yscale = y/scale-y_a
                x_mat = np.array([[xscale**3, xscale**2, xscale, 1]])
                y_mat = np.array([[yscale**3, yscale**2, yscale, 1]])
                result = np.zeros((4, 4))
                for i in range(4):
                    for j in range(4):
                        result[i, j] = img[x_a+1 + i, y_a+1 + j, channel]

                tmp_mat = np.matmul(x_mat,np.matmul(np.linalg.inv(bic_mat), result)).T
                final = int(np.matmul(y_mat, np.matmul(np.linalg.inv(bic_mat), tmp_mat)))
                if final >255:
                    final = 255
                elif final <0:
                    final = 0
                bic_result[x,y,channel] = final


    show(near_result, fig.add_subplot(rows, cols, 2), "nearest image")
    show(bil_result, fig.add_subplot(rows, cols, 3), "bilinear image")
    show(bic_result, fig.add_subplot(rows, cols, 4), "bicubic image")

    plt.show()
    ### interpolation end
    '''

    img_dir = "./example"
    img_name = "DCT2.png"        # 1은 png 2는 bmp

    img = cv2.imread(os.path.join(img_dir, img_name),cv2.IMREAD_GRAYSCALE)
    show(img, fig.add_subplot(rows,cols,1), "original image")
    img = np.float32(img)/255.

    if img.shape[0] %2 != 0:
        img = cv2.copyMakeBorder(img, 0,0,0,1,cv2.BORDER_REFLECT)
    if img.shape[1] %2 != 0:
        img = cv2.copyMakeBorder(img, 0,0,1,0,cv2.BORDER_REFLECT)

    dct = cv2.dct(img)

    dct[dct<0] = 0
    dct = np.uint8((dct)*255)
    show(dct, fig.add_subplot(rows,cols,2), "DCT image")

    # cv2.imshow("asd",dct)
    # cv2.waitKey()
    plt.show()