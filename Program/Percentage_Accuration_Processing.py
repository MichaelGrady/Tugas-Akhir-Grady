# Percentage Error Processing
# Created by : Michael Grady
# Mechanical Engineering AtmaJaya Catholic University of Indonesia
# For bachelor thesis : “The Development of Road Area Detection System Using Computer Vision Technique”

import cv2 
import numpy as np

# GREEN = image 1, RED = image 2, YELLOW = image 1 + image 2
# 4900 Complete Dark, 6700 No Road

loc = "E:\\Grady\\Programming\\Python\\TugasAkhirGrady\\HUMAN VISION\\masked\\8300.jpg" # true image as parameter

# loc2 = "E:\\Grady\\Programming\\Python\\TugasAkhirGrady\\mask_perspective_adaptive\\8300.jpg" # adaptive
loc2 = "E:\\Grady\\Programming\\Python\\TugasAkhirGrady\\mask_perspective_non_adaptive\\8300.jpg" # non-adaptive

img = cv2.imread(loc, cv2.IMREAD_COLOR)
img2 = cv2.imread(loc2, cv2.IMREAD_COLOR)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

# masking parameter
low_mask = np.array([-40, -40, 205])
up_mask = np.array([40,  40, 305])
low_yellow = np.array([20, 231, 225])
up_yellow = np.array([43, 265, 285])
low_green = np.array([31, 225, 225])
up_green = np.array([91, 285, 285])
low_red = np.array([-10, 245, 218])
up_red = np.array([10, 265, 284])


mask = cv2.inRange(hsv, low_mask, up_mask)
mask2 = cv2.inRange(hsv2, low_mask, up_mask)

# blending img
ret,thresh = cv2.threshold(mask, 40, 255, 0)
im,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
mask_change = cv2.drawContours(img, contours, -1, (0,255,0), -1)
ret2,thresh2 = cv2.threshold(mask2, 40, 255, 0)
im2,contours2,hierarchy2 = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
mask_change2 = cv2.drawContours(img2, contours2, -1, (0,0,255), -1)
blendmask = cv2.addWeighted(mask_change, 1, mask_change2, 1, 0.0)
hsv_blend = cv2.cvtColor(blendmask, cv2.COLOR_BGR2HSV)

mask_yellow = cv2.inRange(hsv_blend, low_yellow, up_yellow)
mask_green = cv2.inRange(hsv_blend, low_green, up_green)
mask_red = cv2.inRange(hsv_blend, low_red, up_red)

tot_pixel = img.size/3 
image_dimension = tot_pixel**0.5

# total of green pixel from true images as parameter and red from the image processing
green_true_pixel = np.count_nonzero(mask)
red_true_pixel = np.count_nonzero(mask2)

# total of green, red, and yellow pixel from blending images
green_pixel = np.count_nonzero(mask_green)
red_pixel = np.count_nonzero(mask_red)
yellow_pixel = np.count_nonzero(mask_yellow)
cekgreen = yellow_pixel + green_pixel
cekred = yellow_pixel + red_pixel

true_count = yellow_pixel-(green_pixel + red_pixel)
yellowpergreen = yellow_pixel / green_true_pixel
yellowpergreen_percent = round(yellowpergreen * 100)
accuration = round(true_count * 100 / green_true_pixel)
error_percentage = round(100 - accuration)

print(loc)
print(loc2)
print("Total pixels: " + str(tot_pixel))
print("True Green pixels: " + str(green_true_pixel))
print("True Red pixels: " + str(red_true_pixel))
print("Green pixels: " + str(green_pixel))
print("Red pixels: " + str(red_pixel))
print("Yellow pixels: " + str(yellow_pixel))
print("True pixel count: " + str(true_count))
# print("Yellow / True green: " + str(yellowpergreen_percent) + "%")
# print("Accuration: " + str(accuration) + "%")
# print("Percentage of Error: " + str(error_percentage) + "%")
print(int(green_true_pixel), int(red_true_pixel), int(green_pixel), int(red_pixel), int(yellow_pixel))
print(cekgreen, cekred)

# cv2.imwrite('E:\\Grady\\Programming\\Python\\TugasAkhirGrady\\PE_adaptive\\8650.jpg', blendmask) # adaptive
# cv2.imwrite('E:\\Grady\\Programming\\Python\\TugasAkhirGrady\\PE_nonadaptive\\8650.jpg', blendmask)
# cv2.imwrite('E:\\Grady\\Programming\\Python\\TugasAkhirGrady\\Result Gabungan\\true8000.jpg', mask_change)
# cv2.imwrite('E:\\Grady\\Programming\\Python\\TugasAkhirGrady\\Result Gabungan\\blend8000.jpg', blendmask) # non adaptive
# cv2.imshow("frame", img)
# cv2.imshow("frame2", img2)
# cv2.imshow("green", mask_green)
# cv2.imshow("red", mask_red)
# cv2.imshow("yellow", mask_yellow)

cv2.imshow("blend", blendmask)
cv2.imshow("true", mask_change)
cv2.imshow("adaptive", mask_change2)

cv2.waitKey(0)
cv2.destroyAllWindows()
