import numpy as np
import cv2
from numpy.lib.stride_tricks import as_strided

def sum_abs_diff(image, template, mask=None):
    if mask is None:
        mask = np.ones_like(template)
    window_size = template.shape

    new_img = as_strided(image, shape=(image.shape[0] - window_size[0] + 1, image.shape[1] - window_size[1] + 1,) +
                                           window_size, strides=image.strides * 2)

    sad = ((abs(new_img - template)) * mask)
    sad = sad.sum(axis=-1)
    sad = sad.sum(axis=-1)
    return sad


def disparity(left, right, template, window, option):
    disparity = np.zeros(left.shape, dtype=np.float32)
    
    for row in range(int(template/2), int(left.shape[0]-template/2)):
        tr_min = max(int(row -template/2), 0)
        tr_max = min(int(row+template/2)+1, left.shape[0])
        for col in range(int(template/2), int(left.shape[1]-template/2)):
            tc_min = max(int(col - template/2), 0)
            tc_max = min(int(col + template/2)+1, left.shape[1])
            tpl = left[tr_min:tr_max, tc_min:tc_max].astype(np.float32)
            rc_min = max(int(col - window / 2), 0)
            rc_max = min(int(col + window / 2) + 1, left.shape[1])
            R_strip = right[tr_min:tr_max, rc_min:rc_max].astype(np.float32)

            if option == "SSD":
                error = cv2.matchTemplate(R_strip, tpl, method=cv2.TM_SQDIFF)
            elif option == "SAD":
                error = sum_abs_diff(R_strip, tpl)
            elif option == "NCC":
                error = cv2.matchTemplate(R_strip, tpl, method=cv2.TM_CCORR_NORMED)

            c_tf = max(col - rc_min-template/2, 0)
            dist = np.arange(error.shape[1]) - c_tf
            cost = error + np.abs(dist * 0)
            if option == "NCC":
                _,_,_,max_loc = cv2.minMaxLoc(cost)
                disparity[row, col] = dist[max_loc[0]]
            else:
                _,_,min_loc,_ = cv2.minMaxLoc(cost)
                disparity[row, col] = dist[min_loc[0]]
    return disparity