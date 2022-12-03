import numpy as np
import cv2
from disparity import *


def expand_n_double_disparity(img, levels):
    exp_img = img

    for i in range(0, img.shape[0], 2**(levels-1)):
        for j in range(0, img.shape[1], 2**(levels-1)):
            for k in range(3):
                exp_img[i:i+2**(levels-1), j:j+2**(levels-1), k] = img[i, j, k]

    return exp_img


def prepare(levels):
    #left, right = getOriginalImages()
    left = cv2.imread('Original images/left1.png')
    right = cv2.imread('Original images/right1.png')

    left = expand_n_double_disparity(left, levels)
    right = expand_n_double_disparity(right, levels)

    left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    return left, right


# Stereo matching using SSD
def ssd(levels, template, window):

    # Calculate disparity maps of the left and right images
    left, right = prepare(levels)
    left_to_right_disp = np.abs(disparity(left, right, template, window, "SSD"))
    right_to_left_disp = np.abs(disparity(right, left, template, window, "SSD"))

    # Scale disparity maps
    left_to_right_disp = cv2.normalize(left_to_right_disp, left_to_right_disp, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    right_to_left_disp = cv2.normalize(right_to_left_disp, right_to_left_disp, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return left_to_right_disp, right_to_left_disp


# Stereo matching using SAD
def sad(levels, template, window):
    
    # Calculate disparity maps of the left and right images
    left, right = prepare(levels)
    left_to_right_disp = np.abs(disparity(left, right, template, window, "SAD"))
    right_to_left_disp = np.abs(disparity(right, left, template, window, "SAD"))

    # Scale disparity maps
    left_to_right_disp = cv2.normalize(left_to_right_disp, left_to_right_disp, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    right_to_left_disp = cv2.normalize(right_to_left_disp, right_to_left_disp, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return left_to_right_disp, right_to_left_disp

# Stereo matching using normalized correlation
def ncc(levels, template, window):

    # Calculate disparity maps of the left and right images
    left, right = prepare(levels)
    left_to_right_disp = np.abs(disparity(left, right, template, window, "NCC"))
    right_to_left_disp = np.abs(disparity(right, left, template, window, "NCC"))

    # Scale disparity maps
    left_to_right_disp = cv2.normalize(left_to_right_disp, left_to_right_disp, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    right_to_left_disp = cv2.normalize(right_to_left_disp, right_to_left_disp, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    return left_to_right_disp, right_to_left_disp