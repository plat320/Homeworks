import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import argparse
import copy
import time
from PIL import Image

parser = argparse.ArgumentParser(description="Block_Matching_120200153_LEE_SEONG_HUN")
parser.add_argument('-b', '--block_size', type=int, default=8, help='block size of the block matching algorithm')
parser.add_argument('-s', '--search_range', type=int, default=4, help='search range of the block matching algorithm')
parser.add_argument('-t', '--search_type', type=str, default="Full", help='select type of searching algorithm Full | Three')

args = parser.parse_args()
print(args)

def l2(pred, target):
    tmp = pred.flatten() - target.flatten()
    tmp = (tmp**2).mean()
    return tmp

def show(img, fig, title):
    #### for RGB
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    plt.imsave(title+".png", np.uint8(img))
    # Image.fromarray(np.uint8(img), 'RGB').show(name)
    # plt.imsave(title + ".png", np.uint8(img), cmap='gray')
    # fig.imshow(Image.fromarray(np.uint8(img), 'L'),cmap='gray')
    fig.set_title(title)
    fig.axis("off")

def get_block_ref_pixel(num_H, num_W, args):
    ref_pixel = np.zeros((num_H * num_W, 2))
    for r in range(num_H):
        for c in range(num_W):
            ref_pixel[num_W * r + c, :] = [r*args.block_size, c*args.block_size]

    return ref_pixel.astype(np.int32)

def Full_Search(ref_block, image, start2end, pix, args):
    (start_H, start_W, end_H, end_W) = start2end
    vec = [0, 0]
    min_error = 1000
    for h in range(start_H, end_H + 1):
        for w in range(start_W, end_W + 1):
            error = l2(ref_block, image[h:h + args.block_size, w:w + args.block_size])
            if min_error > error:
                min_error = error
                vec = [h-pix[0], w-pix[1]]

    return vec

def Three_Search(ref_block, image, pix, SR, BS, width, height):
    vec = [0, 0]
    tmp_vec = [0, 0]
    next_pix = pix
    # get nine points
    while True:
        min_error = 1000
        for c in range(-1, 2):
            for r in range(-1, 2):
                h = pix[0] + r * int(SR)
                w = pix[1] + c * int(SR)

                #### exception
                if h < 0 or h + BS >= height:
                    continue
                if w < 0 or w + BS >= width:
                    continue

                error = l2(ref_block, image[h:h + BS, w:w + BS])
                if min_error > error:
                    min_error = error
                    tmp_vec = [h-pix[0], w-pix[1]]
                    next_pix = [h, w]
        pix = next_pix
        vec = [tmp_vec[0] + vec[0], tmp_vec[1] + vec[1]]
        if SR == 1:
            break
        SR = int(SR / 2)


    return vec


if __name__ == '__main__':
    fig = plt.figure()
    rows = 2
    cols = 2


    img_dir = "./input_image/1"
    ref_img_name = "IMG_01830000.png"
    target_img_name = "IMG_01830004.png"
    inter_img_name = "IMG_01830008.png"


    ref_image = cv2.imread(os.path.join(img_dir, ref_img_name),cv2.IMREAD_COLOR)
    target_image = cv2.imread(os.path.join(img_dir, target_img_name),cv2.IMREAD_COLOR)
    inter_image = cv2.imread(os.path.join(img_dir, inter_img_name),cv2.IMREAD_COLOR)
    # show(ref_image, fig.add_subplot(rows,cols,1), "reference image")
    # show(target_image, fig.add_subplot(rows,cols,2), "target image")

    (height, width, channel) = ref_image.shape
    num_H = int(height / args.block_size)
    num_W = int(width / args.block_size)

    #### check condition
    if height % args.block_size != 0 or width % args.block_size != 0:
        width = (width // args.block_size + 1) * args.block_size
        height = (height // args.block_size + 1) * args.block_size
        ref_image = cv2.resize(ref_image, (width,height))
        target_image = cv2.resize(target_image, (width,height))

    #### normalize image
    ref_image = np.float32(ref_image)/255.
    target_image = np.float32(target_image)/255.

    #### Full search
    ref_pixel = get_block_ref_pixel(num_H, num_W, args)     #### column + row * num_width, [height, width]

    vector = copy.deepcopy(ref_pixel)                       #### column + row * num_width, [height, width, motion_h, motion w]
    vector = np.concatenate((vector, np.zeros_like(ref_pixel)), axis=1)



    #### find motion vector
    stime = time.time()
    for idx, pix in enumerate(ref_pixel[:,]):
        ref_block = target_image[pix[0]:pix[0]+args.block_size,     #### for backward mapping
                    pix[1]:pix[1] + args.block_size]                #### BS*BS

        start2end = (max(0, pix[0] - args.search_range),
                     max(0, pix[1] - args.search_range),
                     min(height - args.block_size, pix[0] + args.search_range),
                     min(width - args.block_size, pix[1] + args.search_range))

        if args.search_type == "Full":
            vector[idx, -2:] = Full_Search(ref_block, ref_image, start2end, pix, args)
        else:
            # print(Three_Search(ref_block, ref_image, pix, args.search_range, args.block_size, width, height))
            vector[idx, -2:] = Three_Search(ref_block, ref_image, pix, args.search_range, args.block_size, width, height)

    print("Motion vector estimation finished {:.2f}".format(time.time() - stime))


    recon_image = np.zeros_like(target_image)

    #### image reconstruction
    ref_image = (ref_image * 255).astype(np.uint8)
    for idx, pix in enumerate(vector[:,]):
        [h,w,m_h,m_w] = pix
        h,w,m_h,m_w = int(h),int(w),int(m_h/2), int(m_w/2)
        recon_image[h:h+args.block_size, w:w+args.block_size] = ref_image[
            h+m_h: h+m_h+args.block_size, w+m_w : w+m_w+args.block_size, :
        ]
        # if not(m_h ==0 and m_w == 0):
        #     cv2.arrowedLine(ref_image, (w+m_w, h+m_h), (w, h),  (255), thickness = 1, tipLength = 0.5)

    #### get PSNR
    PSNR = 20 * np.log10(255) - 10 * np.log10(l2(recon_image, inter_image))

    cv2.imwrite("reconstruction_image_-BS_SR_type_PSNR-{}_{}_{}_{:.2f}.png".format(args.block_size, args.search_range, args.search_type, PSNR), recon_image)

    print("reconstruction finished PSNR = {:.4f}".format(PSNR))
    recon_image = (recon_image).astype(np.uint8)



    show(recon_image, fig.add_subplot(rows,cols,3), "reconstruction_image_-BS_SR_type_PSNR-{}_{}_{}_{:.2f}".format(args.block_size, args.search_range, args.search_type, PSNR))

    # plt.show()
