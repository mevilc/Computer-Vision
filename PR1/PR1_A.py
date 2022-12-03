import numpy as np
import cv2
from PR1_A_functions import affine, perspective, display_imgs, error, cv2_args


# main path
path = "Project1/PartA"

# read original images
comp_or_img = cv2.imread(path + '/original/computer.png')
lena_or_img = cv2.imread(path + '/original/lena.png')
mario_or_img = cv2.imread(path + '/original/mario.jpg')
mountain_or_img = cv2.imread(path + '/original/mountain.jpg')
water_or_img = cv2.imread(path + '/original/water.jpg')


# -------------------------AFFINE---------------------------

# METHOD A

# path to each affine .csv file
pts_affine_comp = np.loadtxt(path + "/correspondances/affine/computer.csv", delimiter=',', skiprows=1).astype(np.float32)
pts_affine_lena = np.loadtxt(path + "/correspondances/affine/lena.csv", delimiter=',', skiprows=1).astype(np.float32)
pts_affine_mario = np.loadtxt(path + "/correspondances/affine/mario.csv", delimiter=',', skiprows=1).astype(np.float32)
pts_affine_mountain = np.loadtxt(path + "/correspondances/affine/mountain.csv", delimiter=',', skiprows=1).astype(np.float32)
pts_affine_water = np.loadtxt(path + "/correspondances/affine/water.csv", delimiter=',', skiprows=1).astype(np.float32)


# affine matrices for each original image using method A
affine_trans_comp_A = affine(pts_affine_comp[:3, :])
affine_trans_lena_A = affine(pts_affine_lena[:3, :])
affine_trans_mario_A = affine(pts_affine_mario[:3, :])
affine_trans_mountain_A = affine(pts_affine_mountain[:3, :])
affine_trans_water_A = affine(pts_affine_water[:3, :])

# warp original images with affine matrices from method A
a_trans_comp_A = cv2.warpAffine(comp_or_img, affine_trans_comp_A, (comp_or_img.shape[1]*2, comp_or_img.shape[0]*2))
a_trans_lena_A = cv2.warpAffine(lena_or_img, affine_trans_lena_A, (lena_or_img.shape[1], lena_or_img.shape[0]))
a_trans_mario_A = cv2.warpAffine(mario_or_img, affine_trans_mario_A, (mario_or_img.shape[1]//2, mario_or_img.shape[0]//2))
a_trans_mountain_A = cv2.warpAffine(mountain_or_img, affine_trans_mountain_A, (mountain_or_img.shape[1], mountain_or_img.shape[0]))
a_trans_water_A = cv2.warpAffine(water_or_img, affine_trans_water_A, (water_or_img.shape[1], water_or_img.shape[0]))

# Display transformed images from method A
display_imgs(a_trans_comp_A, 'A', 'Affine')
display_imgs(a_trans_lena_A, 'A', 'Affine')
display_imgs(a_trans_mario_A, 'A', 'Affine')
display_imgs(a_trans_mountain_A, 'A', 'Affine')
display_imgs(a_trans_water_A, 'A', 'Affine')

# error between X1 and openCV matrix using method A
errors_X1_A = []
for i in [[pts_affine_comp, affine_trans_comp_A], [pts_affine_lena, affine_trans_lena_A], 
          [pts_affine_mario, affine_trans_mario_A], [pts_affine_mountain, affine_trans_mountain_A],
          [pts_affine_water, affine_trans_water_A]]:

    inp, out = cv2_args(i[0][:3, :])
    a = cv2.getAffineTransform(inp, out)
    aff_err_A = round(error(i[1], a), 2)
    errors_X1_A.append(aff_err_A)

# METHOD B

# affine matrices for each original image using method B
affine_trans_comp_B = affine(pts_affine_comp[:10, :])
affine_trans_lena_B = affine(pts_affine_lena)
affine_trans_mario_B = affine(pts_affine_mario)
affine_trans_mountain_B = affine(pts_affine_mountain)
affine_trans_water_B = affine(pts_affine_water)

# warp original images with affine matrices from method B
a_trans_comp_B = cv2.warpAffine(comp_or_img, affine_trans_comp_B, (comp_or_img.shape[1]*2, comp_or_img.shape[0]*2))
a_trans_lena_B = cv2.warpAffine(lena_or_img, affine_trans_lena_B, (lena_or_img.shape[1], lena_or_img.shape[0]))
a_trans_mario_B = cv2.warpAffine(mario_or_img, affine_trans_mario_B, (mario_or_img.shape[1]//2, mario_or_img.shape[0]//2))
a_trans_mountain_B = cv2.warpAffine(mountain_or_img, affine_trans_mountain_B, (mountain_or_img.shape[1], mountain_or_img.shape[0]))
a_trans_water_B = cv2.warpAffine(water_or_img, affine_trans_water_B, (water_or_img.shape[1], water_or_img.shape[0]))

# display transformed images using method B
display_imgs(a_trans_comp_B, 'B', 'Affine')
display_imgs(a_trans_lena_B, 'B', 'Affine')
display_imgs(a_trans_mario_B, 'B', 'Affine')
display_imgs(a_trans_mountain_B, 'B', 'Affine')
display_imgs(a_trans_water_B, 'B', 'Affine')


# FUNCTION DOES NOT TAKE MORE THAN 3 ARGS
# error between X1 and openCV matrix using method B
'''
errors_X1_B = []
for i in [[pts_affine_comp, affine_trans_comp_B], [pts_affine_lena, affine_trans_lena_B], 
          [pts_affine_mario, affine_trans_mario_B], [pts_affine_mountain, affine_trans_mountain_B],
          [pts_affine_water, affine_trans_water_B]]:

    inp, out = cv2_args(i[0][:10, :])
    a = cv2.getAffineTransform(inp, out)
    aff_err_B = round(error(i[1], a), 2)
    errors_X1_B.append(aff_err_B)

'''
# -----------------------------PERSPECTIVE-----------------------------------

# METHOD A

# read perspective .csv files
pts_pers_comp = np.loadtxt(path + "/correspondances/perspective/computer.csv", delimiter=',', skiprows=1).astype(np.float32)
pts_pers_lena = np.loadtxt(path + "/correspondances/perspective/lena.csv", delimiter=',', skiprows=1).astype(np.float32)
pts_pers_mario = np.loadtxt(path + "/correspondances/perspective/mario.csv", delimiter=',', skiprows=1).astype(np.float32)
pts_pers_mountain = np.loadtxt(path + "/correspondances/perspective/mountain.csv", delimiter=',', skiprows=1).astype(np.float32)
pts_pers_water = np.loadtxt(path + "/correspondances/perspective/water.csv", delimiter=',', skiprows=1).astype(np.float32)

# pers matrices for each original image using method A
pers_trans_comp_A = perspective(pts_pers_comp[:4, :])
pers_trans_lena_A = perspective(pts_pers_lena[:4, :])
pers_trans_mario_A = perspective(pts_pers_mario[:4, :])
pers_trans_mountain_A = perspective(pts_pers_mountain[:4, :])
pers_trans_water_A = perspective(pts_pers_water[:4, :])


# warp original images with perspective matrices from method A
p_trans_comp_A = cv2.warpPerspective(comp_or_img, pers_trans_comp_A, (comp_or_img.shape[1]*2, comp_or_img.shape[0]*2))
p_trans_lena_A = cv2.warpPerspective(lena_or_img, pers_trans_lena_A, (lena_or_img.shape[1], lena_or_img.shape[0]))
p_trans_mario_A = cv2.warpPerspective(mario_or_img, pers_trans_mario_A, (mario_or_img.shape[1]//2, mario_or_img.shape[0]//2))
p_trans_mountain_A = cv2.warpPerspective(mountain_or_img, pers_trans_mountain_A, (mountain_or_img.shape[1], mountain_or_img.shape[0]))
p_trans_water_A = cv2.warpPerspective(water_or_img, pers_trans_water_A, (water_or_img.shape[1], water_or_img.shape[0]))

# Display transformed images from method A
display_imgs(p_trans_comp_A, 'A', 'Perspective')
display_imgs(p_trans_lena_A, 'A', 'Perspective')
display_imgs(p_trans_mario_A, 'A', 'Perspective')
display_imgs(p_trans_mountain_A, 'A', 'Perspective')
display_imgs(p_trans_water_A, 'A', 'Perspective')

# error between X2 and openCV matrix using method A
errors_X2_A = []
for i in [[pts_pers_comp, pers_trans_comp_A], [pts_pers_lena, pers_trans_lena_A], 
          [pts_pers_mario, pers_trans_mario_A], [pts_pers_mountain, pers_trans_mountain_A],
          [pts_pers_water, pers_trans_water_A]]:

    inp, out = cv2_args(i[0][:4, :])
    a = cv2.getPerspectiveTransform(inp, out)
    pers_err_A = round(error(i[1], a), 2)
    errors_X2_A.append(pers_err_A)


# METHOD B

# pers matrices for each original image using method B
pers_trans_comp_B = perspective(pts_pers_comp[:10, :])
pers_trans_lena_B = perspective(pts_pers_lena)
pers_trans_mario_B = perspective(pts_pers_mario)
pers_trans_mountain_B = perspective(pts_pers_mountain)
pers_trans_water_B = perspective(pts_pers_water)

# warp original images with affine matrices from method B
p_trans_comp_B = cv2.warpPerspective(comp_or_img, pers_trans_comp_B, (comp_or_img.shape[1]*2, comp_or_img.shape[0]*2))
p_trans_lena_B = cv2.warpPerspective(lena_or_img, pers_trans_lena_B, (lena_or_img.shape[1], lena_or_img.shape[0]))
p_trans_mario_B = cv2.warpPerspective(mario_or_img, pers_trans_mario_B, (mario_or_img.shape[1]//2, mario_or_img.shape[0]//2))
p_trans_mountain_B = cv2.warpPerspective(mountain_or_img, pers_trans_mountain_B, (mountain_or_img.shape[1], mountain_or_img.shape[0]))
p_trans_water_B = cv2.warpPerspective(water_or_img, pers_trans_water_B, (water_or_img.shape[1], water_or_img.shape[0]))

# Display transformed images from method B
display_imgs(p_trans_comp_B, 'B', 'Perspective')
display_imgs(p_trans_lena_B, 'B', 'Perspective')
display_imgs(p_trans_mario_B, 'B', 'Perspective')
display_imgs(p_trans_mountain_B, 'B', 'Perspective')
display_imgs(p_trans_water_B, 'B', 'Perspective')

'''
# error between X2 and openCV matrix using method B
errors_X2_B = []
for i in [[pts_pers_comp, pers_trans_comp_B], [pts_pers_lena, pers_trans_lena_B], 
          [pts_pers_mario, pers_trans_mario_B], [pts_pers_mountain, pers_trans_mountain_B],
          [pts_pers_water, pers_trans_water_B]]:

    inp, out = cv2_args(i[0][:4, :])
    a = cv2.getPerspectiveTransform(inp, out)
    pers_err_B = round(error(i[1], a), 2)
    errors_X2_B.append(pers_err_B)
'''


#-------------------------- EXTRA CORRESPONDANCES-----------------------------------

# Affine

affine_trans_comp = affine(pts_affine_comp)
a_trans_comp = cv2.warpAffine(comp_or_img, affine_trans_comp, (comp_or_img.shape[1]*2, comp_or_img.shape[0]*2))
display_imgs(a_trans_comp, 'using extra Affine correspondaces')

# Affine error -- cant get more than 3 inputs to cv2 function

inp, out = cv2_args(pts_affine_comp[:3])
#print(inp, out)
a = cv2.getAffineTransform(inp, out)
aff_err_comp = round(error(affine_trans_comp, a), 2)

# Perspective

pers_trans_comp = perspective(pts_affine_comp)
p_trans_comp = cv2.warpPerspective(comp_or_img, pers_trans_comp, (comp_or_img.shape[1]*2, comp_or_img.shape[0]*2))
display_imgs(p_trans_comp, 'using extra Perspective correspondaces')

# Pers error -- cant get more than 3 inputs to cv2 function

inp, out = cv2_args(pts_pers_comp[:4])
#print(inp, out)
a = cv2.getPerspectiveTransform(inp, out)
pers_err_comp = round(error(pers_trans_comp, a), 2)