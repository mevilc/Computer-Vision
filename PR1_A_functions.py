import numpy as np
import cv2
from numpy.linalg import inv


def error(M1, M2):
    '''
    (array, array) -> float
    M1 - Matrix 1, M2 - Matrix 2

    Returns error between M1 and M2. Error is computed as summation
    of the absolute value of element-wise difference b/w M1 and M2.
    '''
    return np.sum(np.sum(np.abs(M1 - M2)))

def cv2_args(pts):
    '''
    (array) -> array, array
    pts - point correspondances array

    Returns two arrays (inp - input points and out - transformed pts)
    from input points correspondances array.
    '''
    inp = [[]] * len(pts)
    out = [[]] * len(pts)
    for idx, i in enumerate(pts):
        inp[idx] = [i[0], i[1]]
        out[idx] = [i[2], i[3]]
    return np.float32(inp), np.float32(out)

def display_imgs(trans_img, method=None, type=None):
    '''
    (array, str, str) -> None
    trans_img - input image
    method - A or B
    type - minimal/over constrained points

    Displays image.
    '''
    if method == None:
        cv2.imshow(f'Transformed image using extra {type} points', trans_img)
        cv2.waitKey()
        cv2.destroyAllWindows()
    else:
        cv2.imshow(f'{type} -- transformed using method {method}', trans_img)
        cv2.waitKey()
        cv2.destroyAllWindows()


def affine(pts):
    '''
    (array) -> array
    pts: point correspondance array

    Returns the affine transformation matrix from pts.
    '''
    rows = len(pts) * 2
    A = np.float32([np.zeros(6)] * rows)
    b = np.float32([np.zeros(1)] * rows)

    j = 0
    for i in range(0, rows - 1, 2):
        x1, y1 = pts[j, 0], pts[j, 1]
        x1p, y1p = pts[j, 2], pts[j, 3]

        b[i, 0], b[i+1, 0] = x1p, y1p
        A[i, :] = [x1, y1, 1, 0, 0, 0]
        A[i+1, :] = [0, 0, 0, x1, y1, 1]
        j += 1

    ps_inv = inv(A.T @  A)
    a = ps_inv @ A.T
    H = a @ b
    return H.reshape(2, 3)

def perspective(pts):
    '''
    (array) -> array
    pts: point correspondance array

    Returns the pers transformation matrix from pts.
    '''
    rows = len(pts) * 2
    A = np.float32([np.zeros(9)] * rows)
    j = 0
    for i in range(0, rows - 1, 2):
        x1, y1 = pts[j, 0], pts[j, 1]
        x1p, y1p = pts[j, 2], pts[j, 3]

        A[i, :] = [-x1, -y1, -1, 0, 0, 0, x1*x1p, y1*x1p, x1p]
        A[i+1, :] = [0, 0, 0, -x1, -y1, -1, x1*y1p, y1*y1p, y1p]
        j += 1

    U, S, V = np.linalg.svd(A)
    return (V[-1, :] / V[-1, -1]).reshape(3, 3)