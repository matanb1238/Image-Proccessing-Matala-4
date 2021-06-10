import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from time import gmtime, strftime

def disparitySSD(img_l: np.ndarray, img_r: np.ndarray, disp_range: (int, int), k_size: int) -> np.ndarray:
    disparity_map = np.zeros_like(img_l)
    for x in range(0, img_l.shape[0]):
        for y in range(0, img_l.shape[1]):
            min = np.Inf
            i_at_min = 0
            kernel_left = img_l[x-k_size: x+k_size + 1, y-k_size: y+k_size+1]
            for i in range(disp_range[0], disp_range[1]):
                if(k_size < y-i) and (y+i < img_l.shape[1]-k_size) and (x > k_size) and (x < img_l.shape[0]-k_size):
                    # shift right
                    kernel_right = img_r[x-k_size: x+k_size+1, y+i-k_size: y+i+k_size+1]
                    r_m = np.sum(np.square(kernel_left-kernel_right))
                    if r_m < min:
                        min = r_m
                        i_at_min = i
                    # shift left
                    kernel_right = img_r[x-k_size: x+k_size+1, y-i-k_size: y-i+k_size+1]
                    l_m = np.sum(np.square(kernel_left - kernel_right))
                    if l_m < min:
                        min = l_m
                        i_at_min = i
            disparity_map[x, y] = i_at_min
    return disparity_map
    """

    img_l: Left image
    img_r: Right image
    range: Minimun and Maximum disparity range. Ex. (10,80)
    k_size: Kernel size for computing the SSD, kernel.shape = (k_size*2+1,k_size*2+1)

    return: Disparity map, disp_map.shape = Left.shape
    """
    pass

def disparityNC(img_l: np.ndarray, img_r: np.ndarray, disp_range: (int, int), k_size: int) -> np.ndarray:
    disparity_map = np.zeros_like(img_l)
    for x in range(0, img_l.shape[0]):
        for y in range(0, img_l.shape[1]):
            max = -1000000
            i_at_max = 0
            kernel_left = img_l[x-k_size: x+k_size + 1, y-k_size: y+k_size+1]
            norm = np.linalg.norm(kernel_left)
            left_sum_square = np.sqrt(np.sum(np.square(kernel_left - norm)))
            left_sum = kernel_left - norm
            for i in range(disp_range[0], disp_range[1]):
                if (k_size < y - i) and (y + i < img_l.shape[1] - k_size) and (x > k_size) and (x < img_l.shape[0] - k_size):
                    # shift right
                    kernel_right = img_r[x-k_size: x+k_size+1, y+i-k_size: y+i+k_size+1]
                    norm = np.linalg.norm(kernel_right)
                    right_sum_square = np.sqrt(np.sum(np.square(kernel_right - norm)))
                    right_sum = kernel_right - norm
                    ncc = (np.sum(left_sum * right_sum))/(left_sum_square*right_sum_square)
                    if ncc > max:
                        max = ncc
                        i_at_max = i

                    # shift left
                    kernel_right = img_r[x - k_size: x + k_size + 1, y - i - k_size: y - i + k_size + 1]
                    norm = np.linalg.norm(kernel_right)
                    right_sum_square = np.sqrt(np.sum(np.square(kernel_right - norm)))
                    right_sum = kernel_right - norm
                    ncc = (np.sum(left_sum * right_sum))/(left_sum_square*right_sum_square)
                    if ncc > max:
                        max = ncc
                        i_at_max = i
            disparity_map[x, y] = i_at_max
    return disparity_map
    """

    img_l: Left image
    img_r: Right image
    range: Minimun and Maximum disparity range. Ex. (10,80)
    k_size: Kernel size for computing the SSD, kernel.shape = (k_size*2+1,k_size*2+1)

    return: Disparity map, disp_map.shape = Left.shape
    """

def computeHomography(src_pnt: np.ndarray, dst_pnt: np.ndarray) -> (np.ndarray, float):
    # matrix is A in the tirgul
    matrix = np.ndarray(shape=(8, 9))
    for i in range(0, matrix.shape[0]//2):
        x = src_pnt[i][0]
        y = src_pnt[i][1]
        x_tag = dst_pnt[i][0]
        y_tag = dst_pnt[i][1]
        matrix[2*i][0] = x
        matrix[2*i][1] = y
        matrix[2*i][2] = 1
        matrix[2*i][3] = 0
        matrix[2*i][4] = 0
        matrix[2*i][5] = 0
        matrix[2*i][6] = -x_tag*x
        matrix[2*i][7] = -x_tag*y
        matrix[2*i][8] = -x_tag

        matrix[2*i+1][0] = 0
        matrix[2*i+1][1] = 0
        matrix[2*i+1][2] = 0
        matrix[2*i+1][3] = x
        matrix[2*i+1][4] = y
        matrix[2*i+1][5] = 1
        matrix[2*i+1][6] = -y_tag * x
        matrix[2*i+1][7] = -y_tag * y
        matrix[2*i+1][8] = -y_tag
    svd = np.linalg.svd(matrix)
    # get d (vH)
    d = svd[-1]
    # get the last
    vec = d[-1]
    # normalize
    M = vec / vec[-1]
    # reshape to 3, 3
    M = np.reshape(M, (3, 3))
    error = 0
    for i in range(4):
        new_src = np.append(src_pnt[i], 1)
        new_dst = np.append(dst_pnt[i], 1)
        transformed_pnt = M.dot(new_src)
        error += np.sqrt(sum((transformed_pnt/transformed_pnt[-1] - new_dst) ** 2))
    # print(M)
    # print(error)
    return M, error


    """
        Finds the homography matrix, M, that transforms points from src_pnt to dst_pnt.
        returns the homography and the error between the transformed points to their
        destination (matched) points. Error = np.sqrt(sum((M.dot(src_pnt)-dst_pnt)**2))

        src_pnt: 4+ keypoints locations (x,y) on the original image. Shape:[4+,2]
        dst_pnt: 4+ keypoints locations (x,y) on the destenation image. Shape:[4+,2]

        return: (Homography matrix shape:[3,3],
                Homography error)
    """
    pass


def warpImag(src_img: np.ndarray, dst_img: np.ndarray) -> None:
    """
       Displays both images, and lets the user mark 4 or more points on each image. Then calculates the homography and transforms the source image on to the destination image. Then transforms the source image onto the destination image and displays the result.

       src_img: The image that will be 'pasted' onto the destination image.
       dst_img: The image that the source image will be 'pasted' on.

       output:
        None.
    """

    dst_p = []
    fig1 = plt.figure()

    def onclick_1(event):
        x = event.xdata
        y = event.ydata
        print("Loc: {:.0f},{:.0f}".format(x, y))

        plt.plot(x, y, '*r')
        dst_p.append([x, y])

        if len(dst_p) == 4:
            plt.close()
        plt.show()


    def onclick_2(event):
        x = event.xdata
        y = event.ydata
        print("Loc: {:.0f},{:.0f}".format(x, y))

        plt.plot(x, y, '*r')
        src_p.append([x, y])

        if len(src_p) == 4:
            plt.close()
        plt.show()

    # display image 1
    cid = fig1.canvas.mpl_connect('button_press_event', onclick_1)
    plt.imshow(dst_img)
    plt.show()
    dst_p = np.array(dst_p)

    # display image 2
    src_p = []
    fig2 = plt.figure()
    cid = fig2.canvas.mpl_connect('button_press_event', onclick_2)
    plt.imshow(src_img)
    plt.show()
    src_p = np.array(src_p)
    
    ##### Your Code Here ######
    # up_left = [0, 0]
    # up_right = [0, src_img.shape[1]-1]
    # down_left = [src_img.shape[0]-1, 0]
    # down_right = [src_img.shape[0]-1, src_img.shape[1]-1]
    # src_p = np.array([up_left, up_right, down_left, down_right])
    # dst_p = np.array(dst_p)
    h, e = computeHomography(src_p, dst_p)
    proj_src = np.zeros_like(dst_img)
    for y in range(src_img.shape[0]):
        for x in range(src_img.shape[1]):
            src_point = np.array([x, y, 1]).T
            dst_point = h.dot(src_point)
            i = int(dst_point[0]/dst_point[2])
            j = int(dst_point[1]/dst_point[2])
            proj_src[j, i] = src_img[y, x]

    # mask
    mask = proj_src == 0
    canvas = dst_img * mask + (1 - mask) * proj_src

    plt.imshow(canvas)
    plt.show()
    pass
