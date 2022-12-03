import cv2
import numpy as np
from disparity import *

def corners(image, window_size, k_size, k):
    gray = np.float32(image)
    output = cv2.cornerHarris(gray, window_size, k_size, k)
    image[output>1*output.max()] = 255
    return image

def broadcast(image, levels):
    #h, w, c = image.shape

    outputImage = image

    for i in range(0, image.shape[0], 2**(levels-1)):
        for j in range(0, image.shape[1], 2**(levels-1)):
            for k in range(3):
                outputImage[i:i+2**(levels-1), j:j+2**(levels-1), k] = image[i, j, k]

    return outputImage


def prepare(levels):
    left = cv2.imread('Original images/left1.png')
    right = cv2.imread('Original images/right1.png')

    left = broadcast(left, levels)
    right = broadcast(right, levels)

    left = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY)
    right = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY)

    window_size = 3
    k = 0.15

    new_Left = corners(left, window_size, 3, k)
    new_Right = corners(right, window_size, 3, k)
    return new_Left, new_Right


# Stereo matching using SSD
def ssd(levels, template, window):

    left, right = prepare(levels)

    left_to_right_disp = np.abs(disparity(left, right, template, window, "SSD"))
    right_to_left_disp = np.abs(disparity(right, left, template, window, "SSD"))

    left_to_right_disp = cv2.normalize(left_to_right_disp, left_to_right_disp, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    right_to_left_disp = cv2.normalize(right_to_left_disp, right_to_left_disp, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return left_to_right_disp, right_to_left_disp


# Stereo matching using SAD
def sad(levels, template, window):

    left, right = prepare(levels)

    left_to_right_disp = np.abs(disparity(left, right, template, window, "SAD"))
    right_to_left_disp = np.abs(disparity(right, left, template, window, "SAD"))

    left_to_right_disp = cv2.normalize(left_to_right_disp, left_to_right_disp, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    right_to_left_disp = cv2.normalize(right_to_left_disp, right_to_left_disp, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return left_to_right_disp, right_to_left_disp


# Stereo matching using normalized correlation
def ncc(levels, template, window):

    left, right = prepare(levels)

    left_to_right_disp = np.abs(disparity(left, right, template, window, "NCC"))
    right_to_left_disp = np.abs(disparity(right, left, template, window, "NCC"))

    left_to_right_disp = cv2.normalize(left_to_right_disp, left_to_right_disp, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    right_to_left_disp = cv2.normalize(right_to_left_disp, right_to_left_disp, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return left_to_right_disp, right_to_left_disp