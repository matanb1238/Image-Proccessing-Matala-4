# ps2
import os
import numpy as np
from ex4_utils import *
import cv2

def ID(ID: int):
    print(ID)

def displayDepthImage(l_img, r_img, disparity_range=(0, 5), method=disparitySSD):
    p_size = 5
    d_ssd = method(l_img, r_img, disparity_range, p_size)
    plt.matshow(d_ssd)
    plt.colorbar()
    plt.show()


def main():
    ID(323010835)
    ## 1-a
    # Read images
    i = 0
    if i == 0:
        min_r, max_r = 0, 5
    else:
        min_r, max_r = 0, 150
    # L = cv2.imread(os.path.join('input', 'pair%d-L.png' % i), 0) / 255.0
    # R = cv2.imread(os.path.join('input', 'pair%d-R.png' % i), 0) / 255.0
    L = cv2.imread('pair%d-L.png' % i, 0) / 255.0
    R = cv2.imread('pair%d-R.png' % i, 0) / 255.0
    # Display depth SSD
    displayDepthImage(L, R, (min_r, max_r), method=disparitySSD)
    # Display depth NC
    displayDepthImage(L, R, (min_r, max_r), method=disparityNC)

    src = np.array([[279, 552],
                    [372, 559],
                    [362, 472],
                    [277, 469]])
    dst = np.array([[24, 566],
                    [114, 552],
                    [106, 474],
                    [19, 481]])
    h, error = computeHomography(src, dst)
    print(h, error)
    # h_src = Homogeneous(src)
    # pred = h.dot(h_src.T).T
    #
    # pred = unHomogeneous(pred)
    # print(np.sqrt(np.square(pred-dst).mean()))

    # dst = cv2.imread(os.path.join('input', 'billBoard.jpg'), 0) / 255.0
    # src = cv2.imread(os.path.join('input', 'car.jpg'), 0) / 255.0
    dst = cv2.imread('billBoard.jpg', 0) / 255.0
    src = cv2.imread('car.jpg', 0) / 255.0
    warpImag(src, dst)


if __name__ == '__main__':
    main()
