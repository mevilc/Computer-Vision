import cv2
import numpy as np

path = 'Project1/PartB/lena.png'
img = cv2.imread(path)

# QUESTION 1
def convolve(img, K):
    '''
    img (array) - image to convolve
    K (array) - kernel to convolve with

    Returns the convolved image after applying K to img.
    '''
    k_size = K.shape[1]
    out_img_size = (img.shape[0] - k_size + 1, img.shape[1] - k_size + 1)
    n = out_img_size[0]
    conv_img = np.zeros(shape=out_img_size).astype(np.float32)

    # grayscale image
    if len(img.shape) == 2:
        for row in range(n):
            for col in range(n):
                mat = img[row:row + k_size, col : col+ k_size]
                conv_img[row, col] = np.sum(np.multiply(mat, K))

        conv_img /= 255

    # RGB image
    else:
        conv_img = np.zeros(shape=(img.shape[1] - k_size + 1, img.shape[0] - k_size + 1, 3)).astype(np.uint8) # (m, m, 3)
        for row in range(n):
            for col in range(n):
                mat = img[row:row + k_size, col : col+ k_size]
                R_mat, G_mat, B_mat = rgb(mat, k_size)
                print(row, col)
                conv_img[row, col] = [np.sum(R_mat * K), np.sum(G_mat * K), np.sum(B_mat * K)]
    return conv_img

# QUESTION 1 HELPER
def rgb(mat, k_size):
    '''
    mat (array) - image array
    k_size (int) - kernel size

    Returns 3 convolve windows of a kxk kernel on a NxN image
    with only R, G, and B values.
    '''
    R_mat, G_mat, B_mat = np.zeros(shape=(k_size, k_size)), np.zeros(shape=(k_size, k_size)), np.zeros(shape=(k_size, k_size))
    for idx, elem in enumerate(mat):
        for jdx, pixels in enumerate(elem):
            R_mat[idx, jdx] = pixels[0]
            G_mat[idx, jdx] = pixels[1]
            B_mat[idx, jdx] = pixels[2]
    return R_mat, G_mat, B_mat

# QUESTION 2
def Reduce(img):
    '''
    img (array) - image array

    Returns an img reduced by half.
    '''

    # filter
    K = np.float32([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
    filtered_img = convolve(img, K)
    H, W, N = filtered_img.shape # (rows, cols, channels)

    reduced_img = np.zeros(shape=(H//2, W//2, 3)).astype(np.uint8)
    for row in range(0, H-1, 2):
        for col in range(0, W-1, 2):
            reduced_img[row//2, col//2] = filtered_img[row, col]

    return reduced_img


# QUESITON 2 HELPER
def padding(img_to_pad, or_img):
    '''
    img_to_pad (array) - img array
    or_img (array) - img array

    Returns a padded image.
    '''
    
    H, W, N = img_to_pad.shape
    H_, W_, N_ = or_img.shape

    l_r_pad = [np.zeros(shape=(H, W, 3)), img_to_pad, np.zeros(shape=(H, W, 3))]            
    l_r_padded = np.hstack(l_r_pad)
    u_d_pad = [np.zeros(shape=(2 * H + H, 2 * H + H, 3)), l_r_padded, np.zeros(shape=(2 * H + H, 2 * H + H, 3))]
    u_d_padded = np.vstack(u_d_pad)
    l_r_pxls = W_ - W

    # 1530 - 1 = 1529
    padded_img = u_d_padded[(2 * H + H) - (l_r_pxls // 2) : (2 * H + H) - (l_r_pxls // 2) + H + l_r_pxls, 
                            W - (l_r_pxls // 2) : W - (l_r_pxls // 2) + (2 * W + W) - (l_r_pxls // 2) + W + l_r_pxls]
    
    d = [np.zeros(shape=(510, 510, 3)), img_to_pad, np.zeros(shape=(510, 510, 3))]            
    c = np.hstack(d)
    b = [np.zeros(shape=(1530,1530, 3)), c, np.zeros(shape=(1530, 1530, 3))]
    e = np.vstack(b)
    padded_img = e[1529 : 2041, 509 : 1021]
    return padded_img


# QUESTION 3 HELPER
def space(a):
    '''
    a (array) - inp img

    Returns an image twice the size with pixels of value 0
    in between.
    '''
    b = np.zeros(shape=(a.shape[0], a.shape[1] * 2, 3))
    for j in range(a.shape[0]):
        for i in range(a.shape[1]):
            b[j, 2*i] = a[j, i]

    b[:, -1] = b[:, -2]

    for i in range(a.shape[0] * 2):
        if i % 2 != 0:
            b = np.insert(b, i, np.zeros(shape=(1, a.shape[1] * 2, 3)), axis=0)
    return b

# QUESTION 3
def expand(a):
    '''
    a (array) -> image array

    Returns an image twice the size as img with
    pixels interpolated to fill in.
    '''
    c = space(a)
    for i in range(0, c.shape[0], 2): # no of columns
        for j in range(0, c.shape[1]//2, 2):
            c[i, j+1] = [(c[i, j, 0] + c[i, j+2, 0])/2, 
                        (c[i, j, 1] +c[i, j+2, 1])/2, 
                        (c[i, j, 2]+c[i, j+2, 2])/2]
    
    for i in range(0, c.shape[0]//2, 2): # no of columns
        for j in range(0, c.shape[1]):
            c[i+1, j] = [(c[i, j, 0] + c[i+2, j, 0])/2, 
                        (c[i, j, 1] +c[i+2, j, 1])/2, 
                        (c[i, j, 2]+c[i+2, j, 2])/2]

    c[-1] = c[-2]
    return c

# QUESTION 4
def GaussianPyramid(img, n):
    '''
    img (array) -> img array
    n (int) -> no. of levels

    Returns a Gaussian Pyramid of img with levels n.
    '''
    GP = [img]
    gpLevel = img
    for _ in range(n-1):
        gpLevel = Reduce(gpLevel)
        GP.append(gpLevel)
    return GP # [level1, level2....]
    
# QUESTION 5
def LaplacianPyramids(img, n):
    '''
    img (array) -> img array
    n (int) -> no. of levels

    Returns a Laplacian Pyramid of img with levels n.
    '''
    LP = []
    GP = GaussianPyramid(img, n)
    for gLevel in range(len(GP) - 1):
        exp = cv2.resize(GP[gLevel + 1], (GP[gLevel].shape[0], GP[gLevel].shape[0]))
        l_level = GP[gLevel] - exp
        LP.append(l_level)
    LP.append(GP[-1])
    return LP


# QUESTION 6
def Reconstruct(LI, n):
    '''
    LI (array) -> Laplacian Pyramid
    n (int) -> no. of levels

    Returns an original img formed by collapsing a 
    LI of n levels.
    '''
    LI = LI[::-1]
    exp = cv2.resize(LI[0], (LI[1].shape[0], LI[1].shape[0]))
    o_img = [exp + LI[1]]
    for l_level in range(n-2):
        exp = cv2.resize(o_img[-1], (LI[l_level + 2].shape[0], LI[l_level + 2].shape[0]))
        o_img.append(exp + LI[l_level + 2])
    return np.array(o_img[-1])


# QUESTION 7
img1 = cv2.imread('apple.jpeg')
img2 = cv2.imread('orange.jpeg')
LA = LaplacianPyramids(img1)
LB = LaplacianPyramids(img2)