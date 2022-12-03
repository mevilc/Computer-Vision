import cv2, numpy as np

# Averaging is performed in the neighborhood to fill the gaps (zeroes)
def averaging(left, right):
    K_shape = (5, 5)
    K = np.ones(K_shape, np.float32) / 5
    left = cv2.filter2D(left, -1, K)
    right = cv2.filter2D(right, -1, K)
    return left, right