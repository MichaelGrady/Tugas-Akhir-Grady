# Offline Processing
# Created by : Michael Grady
# Mechanical Engineering AtmaJaya Catholic University of Indonesia
# For bachelor thesis : “The Development of Road Area Detection System Using Computer Vision Technique”

import glob
import cv2
import xlrd
import xlwt
from openpyxl import load_workbook
import time
import numpy as np
import threading
import matplotlib.pyplot as plt

# PER-SATUAN yang diubah : LOC, LDR_VAL, IMG_LOCATION

def data_showing():
    global frame_val
    global ldr_val

    loc = (".xlsx") # every data included from kamis 08.00 - kamis 25 april 17.22

    ####### SATUAN #######    
    # loc = (".xlsx")  # kamis 11 april 08.00
    # loc = (".xlsx")  # kamis 11 april 10.41
    # loc = (".xlsx")  # kamis 11 april 13.30
    # loc = (".xlsx")  # kamis 11 april 15.31
    # loc = (".xlsx")  # kamis 11 april 18.00 
    # loc = (".xlsx")  # kamis 25 april mendung 14.46
    # loc = (".xlsx")  # kamis 25 april 16.37
    # loc = (".xlsx")  # kamis 25 april 17.22
    #######################

    book = load_workbook(loc) 
    sheet = book["Sheet1"]

    for row in sheet.rows:
        # frame_val = row[2].value
        ldr_val = row[3].value
        # ldr_val = row[2].value # SATUAN
        yield ldr_val 

def image_processing():
    # DIST MTX KAMERA LOGITECH HD C270.
    dist = np.float64([[-0.131304517,2.53233311, 0.00159093065, -0.00267024626,-19.7256640]])

    mtx = np.float64([   [1880.88010, 0.            ,  621.613745],
                     [0.          , 1897.97466  ,  287.104560],
                     [0.          , 0.            ,  1.          ]    ])

    img_location = glob.glob('E:\\Grady\\Programming\\Python\\TugasAkhirGrady\\Gabungan Atma\\Frame Gabungan Jalanan Atma\\*.jpg') # semuanya

    ####### SATUAN #######    
    # img_location = glob.glob('*.jpg') # Kamis 11 april 08.00
    # img_location = glob.glob('*.jpg') # Kamis 11 april 10.41
    # img_location = glob.glob('*.jpg') # Kamis 11 april 13.30
    # img_location = glob.glob('*.jpg') # Kamis 11 april 15.31
    # img_location = glob.glob('*.jpg') # Kamis 11 april 18.00
    # img_location = glob.glob('*.jpg') # Kamis 25 april mendung 14.46
    # img_location = glob.glob('*.jpg') # Kamis 25 april 16.37
    # img_location = glob.glob('*.jpg') # Kamis 25 april 17.22
    ######################
    i = 1
    img_loc = sorted(img_location, key = len)
    previous_function = data_showing()
    font = cv2.FONT_HERSHEY_SIMPLEX

    for img in img_loc:
        data_yield = next(previous_function) 
        print('%s...' % img) # Print path, frame, LDR

        img = cv2.imread(img, 1)

        # undistortion
        h,  w = img.shape[:2]
        newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
        x,y,w,h = roi
        dst = dst[y:y+h, x:x+w]
        # resizing    
        resize = cv2.resize(dst, (450, 450), interpolation = cv2.INTER_LINEAR) # original
        resize_calibrated = cv2.resize(dst, (450, 450), interpolation = cv2.INTER_LINEAR)
        resize_uncalibrated = cv2.resize(img, (450, 450), interpolation = cv2.INTER_LINEAR)
        
        frame = cv2.GaussianBlur(resize, (5, 5), 0) # adaptive
        frame2 = cv2.GaussianBlur(resize, (5, 5), 0) # HSV FIXED
        frame3 = cv2.GaussianBlur(resize, (5, 5), 0) # show

        # Perspective Transform
        pts1= np.float32([[0, 450],[450, 450],[30, 225],[420,225]])
        pts2 = np.float32([[0, 0], [450, 0], [0, 450], [450, 450]])
        cv2.circle(frame, (pts1[0,0],pts1[0,1]), 5, (0, 255, 255), -1) 
        cv2.circle(frame, (pts1[1,0],pts1[1,1]), 5, (0, 255, 255), -1) 
        cv2.circle(frame, (pts1[2,0],pts1[2,1]), 5, (0, 255, 255), -1)
        cv2.circle(frame, (pts1[3,0],pts1[3,1]), 5, (0, 255, 255), -1)

        cv2.circle(frame3, (pts1[0,0],pts1[0,1]), 5, (0, 255, 255), -1) 
        cv2.circle(frame3, (pts1[1,0],pts1[1,1]), 5, (0, 255, 255), -1) 
        cv2.circle(frame3, (pts1[2,0],pts1[2,1]), 5, (0, 255, 255), -1)
        cv2.circle(frame3, (pts1[3,0],pts1[3,1]), 5, (0, 255, 255), -1)

        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        result = cv2.warpPerspective(frame, matrix, (450,450))

        # show
        matrix2 = cv2.getPerspectiveTransform(pts1, pts2)
        result2 = cv2.warpPerspective(frame3, matrix2, (450,450))

        flipResult = cv2.flip(result, -1)
        flipResult2 = cv2.flip(result, -1)
        flip_original = cv2.flip( flipResult, 1 ) # original
        flip = cv2.flip( flipResult, 1 ) # adaptive
        flip2 = cv2.flip( flipResult, 1 ) # fix
        flip3 = cv2.flip( flipResult2, 1 ) # show

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV) # HSV FIXED
        hsv_perspective = cv2.cvtColor(flip, cv2.COLOR_BGR2HSV)
        hsv_perspective2 = cv2.cvtColor(flip, cv2.COLOR_BGR2HSV)

        ########## MASKING STORAGE ###########
        # HSV FIXED
        low_gray = np.array([22, -12, 104])
        up_gray = np.array([62, 28, 204])
        low_white = np.array([46, -12, 108])
        up_white = np.array([86,  28, 208])
        low_other = np.array([67, 5, 54])
        up_other = np.array([107, 45, 154])

        # LDR X - X # < 11
        low_gray_X = np.array([22, -11,  87])
        up_gray_X = np.array([62,  29, 187])
        low_white_X = np.array([63, -13,  92])
        up_white_X = np.array([103,  27, 192])
        low_other_X = np.array([40, 235, -43])
        up_other_X = np.array([80, 275,  57])

        # LDR XX - XX # 12-20
        low_gray_XX = np.array([22, -12, 104])
        up_gray_XX = np.array([62, 28, 204])
        low_white_XX = np.array([46, -12, 108])
        up_white_XX = np.array([86,  28, 208])
        low_other2_XX = np.array([-30, -30,  17])
        up_other2_XX = np.array([30,  30, 137])
        low_other_XX = np.array([67, 5, 54])
        up_other_XX = np.array([107, 45, 154])
        
        # LDR XXX - XXX # 21-25
        low_gray_XXX = np.array([23, -7, 87])
        up_gray_XXX = np.array([63,  33, 187])
        low_white_XXX = np.array([46, -9, 70])
        up_white_XXX = np.array([86,  31, 170])
        low_other_XXX = np.array([63, -13, 106])
        up_other_XXX = np.array([103, 27, 206])

         # LDR XXXX - XXXX # 26-30
        low_gray_XXXX = np.array([23, -8, 105])
        up_gray_XXXX = np.array([63, 32, 205])
        low_white_XXXX = np.array([50, -20, 100])
        up_white_XXXX = np.array([110, 40, 190])
        low_other_XXXX = np.array([63, -13, 106])
        up_other_XXXX = np.array([103, 27, 206])
        low_other2_XXXX = np.array([40, -1, -8])
        up_other2_XXXX = np.array([100,  59, 112])

         # LDR XXXXX - XXXXX # 31-43
        low_gray_XXXXX = np.array([20, -23,  43])
        up_gray_XXXXX = np.array([80,  37, 163])
        low_white_XXXXX = np.array([36,  -1, -16])
        up_white_XXXXX = np.array([96,  59, 104])
        low_other2_XXXXX = np.array([79, -8, 23])
        up_other2_XXXXX = np.array([146,  76, 143])
        low_other_XXXXX = np.array([53, -18, 44])
        up_other_XXXXX = np.array([123,  42, 164])

         # LDR XXXXXX - XXXXXX # 44-53
        low_gray_XXXXXX = np.array([12, 0, -22])
        up_gray_XXXXXX = np.array([96, 64, 103])
        low_white_XXXXXX = np.array([36,  -1, -16])
        up_white_XXXXXX = np.array([96,  59, 104])
        low_other_XXXXXX = np.array([73, -18, 93])
        up_other_XXXXXX = np.array([133,  42, 213])

        # LDR X7 - X7 # 54-62
        low_gray_X7 = np.array([86, -2, 79])
        up_gray_X7 = np.array([146,  58, 199])
        low_white_X7 = np.array([50, -20, 100])
        up_white_X7 = np.array([110, 40, 190])
        low_other_X7 = np.array([63, -13, 106])
        up_other_X7 = np.array([103, 27, 206])
        low_other2_X7 = np.array([53,  -7, -15])
        up_other2_X7 = np.array([113,  53, 105])

        # LDR X8 - X8 # 63-97
        low_gray_X8 = np.array([53, -23,  78])
        up_gray_X8 = np.array([113, 37, 198])
        low_white_X8 = np.array([81, -13,  91])
        up_white_X8 = np.array([141,  47, 211])
        low_other_X8 = np.array([80, 39, 87])
        up_other_X8 = np.array([140, 99, 207])

        # LDR X9 - X9 # 98-126
        low_gray_X9 = np.array([82, -10, 127])
        up_gray_X9 = np.array([142, 50, 247])
        low_white_X9 = np.array([73, -19, 106])
        up_white_X9 = np.array([133,  41, 226])
        low_other_X9 = np.array([80, 18, 36])
        up_other_X9 = np.array([140, 78, 156])

         # LDR XXXXY - XXXXY # > 127
        low_gray_XXXXY = np.array([96, 59, 63])
        up_gray_XXXXY = np.array([136,  99, 163])
        low_white_XXXXY = np.array([96, 39, 45])
        up_white_XXXXY = np.array([136,  79, 145])
        low_other_XXXXY = np.array([85, 118,  -2])
        up_other_XXXXY = np.array([130, 190,  98])

        ############# WITHOUT PERSPECTIVE TRANSFORM ################
        # Image Processing LDR FIXED HSV FIXED #
        while data_yield >= 0  :
            # masking berdasarkan warna
            mask_gray = cv2.inRange(hsv2, low_gray, up_gray)
            mask_white = cv2.inRange(hsv2, low_white, up_white)
            mask_other = cv2.inRange(hsv2, low_other, up_other)
            mask = mask_gray + mask_white + mask_other
            mask_all = cv2.rectangle(mask,(450,0),(0,175),(0,255,0),-1)

            kernel = np.ones((8,8),np.uint8)
            erosion = cv2.erode(mask_all, kernel, iterations = 1) # experimental
            dilation = cv2.dilate(mask_all, kernel, iterations = 1) # experimental
            ret,thresh = cv2.threshold(mask_all, 40, 255, 0)
            im2,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            if len(contours) != 0 :
                c = max(contours, key = cv2.contourArea)
                cv2.drawContours(frame2, [c], -1, (0, 255, 0), -1)
                cv2.drawContours(frame2, [c], -1, (0, 0, 255), 1)
                M = cv2.moments(c)

                try :
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    center = (cx,cy)
                    extTop = tuple(c[c[:, :, 1].argmin()][0])
                    extBot = tuple(c[c[:, :, 1].argmax()][0])
                    extLeft = tuple(c[c[:, :, 0].argmin()][0])
                    extRight = tuple(c[c[:, :, 0].argmax()][0])
                    cv2.circle(frame2, (cx,cy), 5, (0, 0, 255), -1)
                    cv2.circle(frame2, extLeft, 5, (0, 0, 255), -1) 
                    cv2.circle(frame2, extRight, 5, (0, 0, 255), -1)

                except ZeroDivisionError :
                    print("No Road Detected")
            break

        # Image Processing X - X #
        while data_yield <= 11 :
            # masking berdasarkan warna
            mask_gray = cv2.inRange(hsv, low_gray_X, up_gray_X)
            mask_white = cv2.inRange(hsv, low_white_X, up_white_X)
            mask_other = cv2.inRange(hsv, low_other_X, up_other_X)
            mask = mask_gray + mask_white + mask_other
            mask_all = cv2.rectangle(mask,(450,0),(0,175),(0,255,0),-1)

            kernel = np.ones((8,8),np.uint8)
            erosion = cv2.erode(mask_all, kernel, iterations = 1) # experimental
            dilation = cv2.dilate(mask_all, kernel, iterations = 1) # experimental
            ret,thresh = cv2.threshold(mask_all, 40, 255, 0)
            im2,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            if len(contours) != 0 :
                c = max(contours, key = cv2.contourArea)
                cv2.drawContours(frame, [c], -1, (0, 255, 0), -1)
                cv2.drawContours(frame, [c], -1, (0, 0, 255), 1)
                M = cv2.moments(c)
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                center = (cx,cy)
                extTop = tuple(c[c[:, :, 1].argmin()][0])
                extBot = tuple(c[c[:, :, 1].argmax()][0])
                extLeft = tuple(c[c[:, :, 0].argmin()][0])
                extRight = tuple(c[c[:, :, 0].argmax()][0])
                cv2.circle(frame, (cx,cy), 5, (0, 0, 255), -1)
                cv2.circle(frame, extLeft, 5, (0, 0, 255), -1) 
                cv2.circle(frame, extRight, 5, (0, 0, 255), -1)

            break

        # Image Processing XX - XX #  
        while 12 <= data_yield <= 20 :
            # masking berdasarkan warna
            mask_gray = cv2.inRange(hsv, low_gray_XX, up_gray_XX)
            mask_white = cv2.inRange(hsv, low_white_XX, up_white_XX)
            mask_other2 = cv2.inRange(hsv, low_other2_XX, up_other2_XX)
            mask_other = cv2.inRange(hsv, low_other_XX, up_other_XX)
            mask = mask_gray + mask_white + mask_other
            mask_all = cv2.rectangle(mask,(450,0),(0,175),(0,255,0),-1)

            kernel = np.ones((8,8),np.uint8)
            erosion = cv2.erode(mask_all, kernel, iterations = 1) # experimental
            dilation = cv2.dilate(mask_all, kernel, iterations = 1) # experimental
            ret,thresh = cv2.threshold(mask_all, 40, 255, 0)
            im2,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            if len(contours) != 0 :
                c = max(contours, key = cv2.contourArea)
                cv2.drawContours(frame, [c], -1, (0, 255, 0), -1)
                cv2.drawContours(frame, [c], -1, (0, 0, 255), 1)
                M = cv2.moments(c)

                try :
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    center = (cx,cy)
                    extTop = tuple(c[c[:, :, 1].argmin()][0])
                    extBot = tuple(c[c[:, :, 1].argmax()][0])
                    extLeft = tuple(c[c[:, :, 0].argmin()][0])
                    extRight = tuple(c[c[:, :, 0].argmax()][0])
                    cv2.circle(frame, (cx,cy), 5, (0, 0, 255), -1)
                    cv2.circle(frame, extLeft, 5, (0, 0, 255), -1) 
                    cv2.circle(frame, extRight, 5, (0, 0, 255), -1)

                except ZeroDivisionError :
                    print("No Road Detected")

            break

        # Image Processing XXX - XXX #  
        while 21 <= data_yield <= 25 :
            # masking berdasarkan warna
            mask_gray = cv2.inRange(hsv, low_gray_XXX, up_gray_XXX)
            mask_white = cv2.inRange(hsv, low_white_XXX, up_white_XXX)
            mask_other = cv2.inRange(hsv, low_other_XXX, up_other_XXX)
            mask = mask_gray + mask_white + mask_other
            mask_all = cv2.rectangle(mask,(450,0),(0,175),(0,255,0),-1)

            kernel = np.ones((8,8),np.uint8)
            erosion = cv2.erode(mask_all, kernel, iterations = 1) # experimental
            dilation = cv2.dilate(mask_all, kernel, iterations = 1) # experimental
            ret,thresh = cv2.threshold(mask_all, 40, 255, 0)
            im2,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            if len(contours) != 0 :
                c = max(contours, key = cv2.contourArea)
                cv2.drawContours(frame, [c], -1, (0, 255, 0), -1)
                cv2.drawContours(frame, [c], -1, (0, 0, 255), 1)
                M = cv2.moments(c)
                try :
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    center = (cx,cy)
                    extTop = tuple(c[c[:, :, 1].argmin()][0])
                    extBot = tuple(c[c[:, :, 1].argmax()][0])
                    extLeft = tuple(c[c[:, :, 0].argmin()][0])
                    extRight = tuple(c[c[:, :, 0].argmax()][0])
                    cv2.circle(frame, (cx,cy), 5, (0, 0, 255), -1)
                    cv2.circle(frame, extLeft, 5, (0, 0, 255), -1) 
                    cv2.circle(frame, extRight, 5, (0, 0, 255), -1)

                except ZeroDivisionError :
                    print("No Road Detected")

            break

        # Image Processing XXXX - XXXX #  
        while 26 <= data_yield <= 30 :
            # masking berdasarkan warna
            mask_gray = cv2.inRange(hsv, low_gray_XXXX, up_gray_XXXX)
            mask_white = cv2.inRange(hsv, low_white_XXXX, up_white_XXXX)
            mask_other = cv2.inRange(hsv, low_other_XXXX, up_other_XXXX)
            mask_other2 = cv2.inRange(hsv, low_other2_XXXX, up_other2_XXXX)
            mask = mask_gray + mask_white + mask_other + mask_other2
            mask_all = cv2.rectangle(mask,(450,0),(0,175),(0,255,0),-1)

            kernel = np.ones((8,8),np.uint8)
            erosion = cv2.erode(mask_all, kernel, iterations = 1) # experimental
            dilation = cv2.dilate(mask_all, kernel, iterations = 1) # experimental
            ret,thresh = cv2.threshold(mask_all, 40, 255, 0)
            im2,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            if len(contours) != 0 :
                c = max(contours, key = cv2.contourArea)
                cv2.drawContours(frame, [c], -1, (0, 255, 0), -1)
                cv2.drawContours(frame, [c], -1, (0, 0, 255), 1)
                M = cv2.moments(c)
                try :
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    center = (cx,cy)
                    extTop = tuple(c[c[:, :, 1].argmin()][0])
                    extBot = tuple(c[c[:, :, 1].argmax()][0])
                    extLeft = tuple(c[c[:, :, 0].argmin()][0])
                    extRight = tuple(c[c[:, :, 0].argmax()][0])
                    cv2.circle(frame, (cx,cy), 5, (0, 0, 255), -1)
                    cv2.circle(frame, extLeft, 5, (0, 0, 255), -1) 
                    cv2.circle(frame, extRight, 5, (0, 0, 255), -1)

                except ZeroDivisionError :
                    print("No Road Detected")

            break

        # Image Processing XXXXX - XXXXX #  
        while 31 <= data_yield <= 43 :
            # masking berdasarkan warna
            mask_gray = cv2.inRange(hsv, low_gray_XXXXX, up_gray_XXXXX)
            mask_white = cv2.inRange(hsv, low_white_XXXXX, up_white_XXXXX)
            mask_other = cv2.inRange(hsv, low_other_XXXXX, up_other_XXXXX)
            mask_other2 = cv2.inRange(hsv, low_other2_XXXXX, up_other2_XXXXX)
            mask = mask_gray + mask_white + mask_other + mask_other2
            mask_all = cv2.rectangle(mask,(450,0),(0,175),(0,255,0),-1)

            kernel = np.ones((8,8),np.uint8)
            erosion = cv2.erode(mask_all, kernel, iterations = 1) # experimental
            dilation = cv2.dilate(mask_all, kernel, iterations = 1) # experimental
            ret,thresh = cv2.threshold(mask_all, 40, 255, 0)
            im2,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            if len(contours) != 0 :
                c = max(contours, key = cv2.contourArea)
                cv2.drawContours(frame, [c], -1, (0, 255, 0), -1)
                cv2.drawContours(frame, [c], -1, (0, 0, 255), 1)
                M = cv2.moments(c)
                try :
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    center = (cx,cy)
                    extTop = tuple(c[c[:, :, 1].argmin()][0])
                    extBot = tuple(c[c[:, :, 1].argmax()][0])
                    extLeft = tuple(c[c[:, :, 0].argmin()][0])
                    extRight = tuple(c[c[:, :, 0].argmax()][0])
                    cv2.circle(frame, (cx,cy), 5, (0, 0, 255), -1)
                    cv2.circle(frame, extLeft, 5, (0, 0, 255), -1) 
                    cv2.circle(frame, extRight, 5, (0, 0, 255), -1)

                except ZeroDivisionError :
                    print("No Road Detected")

            break

        # Image Processing XXXXY - XXXXY #  
        while data_yield >= 127 :
            # masking berdasarkan warna
            mask_gray = cv2.inRange(hsv, low_gray_XXXXY, up_gray_XXXXY)
            mask_white = cv2.inRange(hsv, low_white_XXXXY, up_white_XXXXY)
            mask_other = cv2.inRange(hsv, low_other_XXXXY, up_other_XXXXY)
            mask = mask_gray + mask_white + mask_other
            mask_all = cv2.rectangle(mask,(450,0),(0,175),(0,255,0),-1)

            kernel = np.ones((8,8),np.uint8)
            erosion = cv2.erode(mask_all, kernel, iterations = 1) # experimental
            dilation = cv2.dilate(mask_all, kernel, iterations = 1) # experimental
            ret,thresh = cv2.threshold(mask_all, 40, 255, 0)
            im2,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            if len(contours) != 0 :
                c = max(contours, key = cv2.contourArea)
                cv2.drawContours(frame, [c], -1, (0, 255, 0), -1)
                cv2.drawContours(frame, [c], -1, (0, 0, 255), 1)
                M = cv2.moments(c)
                try :
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    center = (cx,cy)
                    extTop = tuple(c[c[:, :, 1].argmin()][0])
                    extBot = tuple(c[c[:, :, 1].argmax()][0])
                    extLeft = tuple(c[c[:, :, 0].argmin()][0])
                    extRight = tuple(c[c[:, :, 0].argmax()][0])
                    cv2.circle(frame, (cx,cy), 5, (0, 0, 255), -1)
                    cv2.circle(frame, extLeft, 5, (0, 0, 255), -1) 
                    cv2.circle(frame, extRight, 5, (0, 0, 255), -1)

                except ZeroDivisionError :
                    print("No Road Detected")

            break

        # Image Processing XXXXXX - XXXXXX #  
        while 44 <= data_yield <= 53 :
            # masking berdasarkan warna
            mask_gray = cv2.inRange(hsv, low_gray_XXXXXX, up_gray_XXXXXX)
            mask_white = cv2.inRange(hsv, low_white_XXXXXX, up_white_XXXXXX)
            mask_other = cv2.inRange(hsv, low_other_XXXXXX, up_other_XXXXXX)
            mask = mask_gray + mask_white + mask_other
            mask_all = cv2.rectangle(mask,(450,0),(0,175),(0,255,0),-1)

            kernel = np.ones((8,8),np.uint8)
            erosion = cv2.erode(mask_all, kernel, iterations = 1) # experimental
            dilation = cv2.dilate(mask_all, kernel, iterations = 1) # experimental
            ret,thresh = cv2.threshold(mask_all, 40, 255, 0)
            im2,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            if len(contours) != 0 :
                c = max(contours, key = cv2.contourArea)
                cv2.drawContours(frame, [c], -1, (0, 255, 0), -1)
                cv2.drawContours(frame, [c], -1, (0, 0, 255), 1)
                M = cv2.moments(c)
                try :
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    center = (cx,cy)
                    extTop = tuple(c[c[:, :, 1].argmin()][0])
                    extBot = tuple(c[c[:, :, 1].argmax()][0])
                    extLeft = tuple(c[c[:, :, 0].argmin()][0])
                    extRight = tuple(c[c[:, :, 0].argmax()][0])
                    cv2.circle(frame, (cx,cy), 5, (0, 0, 255), -1)
                    cv2.circle(frame, extLeft, 5, (0, 0, 255), -1) 
                    cv2.circle(frame, extRight, 5, (0, 0, 255), -1)

                except ZeroDivisionError :
                    print("No Road Detected")

            break 

        # Image Processing X7 - X7 #  
        while 54 <= data_yield <= 62 :
            # masking berdasarkan warna
            mask_gray = cv2.inRange(hsv, low_gray_X7, up_gray_X7)
            mask_white = cv2.inRange(hsv, low_white_X7, up_white_X7)
            mask_other = cv2.inRange(hsv, low_other_X7, up_other_X7)
            mask_other2 = cv2.inRange(hsv, low_other2_X7, up_other2_X7)
            mask = mask_gray + mask_white + mask_other + mask_other2
            mask_all = cv2.rectangle(mask,(450,0),(0,175),(0,255,0),-1)

            kernel = np.ones((8,8),np.uint8)
            erosion = cv2.erode(mask_all, kernel, iterations = 1) # experimental
            dilation = cv2.dilate(mask_all, kernel, iterations = 1) # experimental
            ret,thresh = cv2.threshold(mask_all, 40, 255, 0)
            im2,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            if len(contours) != 0 :
                c = max(contours, key = cv2.contourArea)
                cv2.drawContours(frame, [c], -1, (0, 255, 0), -1)
                cv2.drawContours(frame, [c], -1, (0, 0, 255), 1)
                M = cv2.moments(c)
                try :
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    center = (cx,cy)
                    extTop = tuple(c[c[:, :, 1].argmin()][0])
                    extBot = tuple(c[c[:, :, 1].argmax()][0])
                    extLeft = tuple(c[c[:, :, 0].argmin()][0])
                    extRight = tuple(c[c[:, :, 0].argmax()][0])
                    cv2.circle(frame, (cx,cy), 5, (0, 0, 255), -1)
                    cv2.circle(frame, extLeft, 5, (0, 0, 255), -1) 
                    cv2.circle(frame, extRight, 5, (0, 0, 255), -1)

                except ZeroDivisionError :
                    print("No Road Detected")

            break                

        # Image Processing X8 - X8 #  
        while 63 <= data_yield <= 97 :
            # masking berdasarkan warna
            mask_gray = cv2.inRange(hsv, low_gray_X8, up_gray_X8)
            mask_white = cv2.inRange(hsv, low_white_X8, up_white_X8)
            mask_other = cv2.inRange(hsv, low_other_X8, up_other_X8)
            mask = mask_gray + mask_white + mask_other
            mask_all = cv2.rectangle(mask,(450,0),(0,175),(0,255,0),-1)

            kernel = np.ones((8,8),np.uint8)
            erosion = cv2.erode(mask_all, kernel, iterations = 1) # experimental
            dilation = cv2.dilate(mask_all, kernel, iterations = 1) # experimental
            ret,thresh = cv2.threshold(mask_all, 40, 255, 0)
            im2,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            if len(contours) != 0 :
                c = max(contours, key = cv2.contourArea)
                cv2.drawContours(frame, [c], -1, (0, 255, 0), -1)
                cv2.drawContours(frame, [c], -1, (0, 0, 255), 1)
                M = cv2.moments(c)
                try :
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    center = (cx,cy)
                    extTop = tuple(c[c[:, :, 1].argmin()][0])
                    extBot = tuple(c[c[:, :, 1].argmax()][0])
                    extLeft = tuple(c[c[:, :, 0].argmin()][0])
                    extRight = tuple(c[c[:, :, 0].argmax()][0])
                    cv2.circle(frame, (cx,cy), 5, (0, 0, 255), -1)
                    cv2.circle(frame, extLeft, 5, (0, 0, 255), -1) 
                    cv2.circle(frame, extRight, 5, (0, 0, 255), -1)

                except ZeroDivisionError :
                    print("No Road Detected")

            break 

        # Image Processing X9 - X9 #  
        while 98 <= data_yield <= 126 :
            # masking berdasarkan warna
            mask_gray = cv2.inRange(hsv, low_gray_X9, up_gray_X9)
            mask_white = cv2.inRange(hsv, low_white_X9, up_white_X9)
            mask_other = cv2.inRange(hsv, low_other_X9, up_other_X9)
            mask = mask_gray + mask_white + mask_other
            mask_all = cv2.rectangle(mask,(450,0),(0,175),(0,255,0),-1)

            kernel = np.ones((8,8),np.uint8)
            erosion = cv2.erode(mask_all, kernel, iterations = 1) # experimental
            dilation = cv2.dilate(mask_all, kernel, iterations = 1) # experimental
            ret,thresh = cv2.threshold(mask_all, 40, 255, 0)
            im2,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            if len(contours) != 0 :
                c = max(contours, key = cv2.contourArea)
                cv2.drawContours(frame, [c], -1, (0, 255, 0), -1)
                cv2.drawContours(frame, [c], -1, (0, 0, 255), 1)
                M = cv2.moments(c)
                try :
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    center = (cx,cy)
                    extTop = tuple(c[c[:, :, 1].argmin()][0])
                    extBot = tuple(c[c[:, :, 1].argmax()][0])
                    extLeft = tuple(c[c[:, :, 0].argmin()][0])
                    extRight = tuple(c[c[:, :, 0].argmax()][0])
                    cv2.circle(frame, (cx,cy), 5, (0, 0, 255), -1)
                    cv2.circle(frame, extLeft, 5, (0, 0, 255), -1) 
                    cv2.circle(frame, extRight, 5, (0, 0, 255), -1)

                except ZeroDivisionError :
                    print("No Road Detected")

            break 

        ########## WITH PERSPECTIVE TRANSFORM ##########
        # Image Processing LDR FIXED HSV FIXED #
        while data_yield >= 0  :
            # mask perspective
            mask_gray_perspective = cv2.inRange(hsv_perspective2, low_gray, up_gray)
            mask_white_perspective = cv2.inRange(hsv_perspective2, low_white, up_white)
            mask_other_perspective = cv2.inRange(hsv_perspective2, low_other, up_other)
            mask_perspective= mask_gray_perspective + mask_white_perspective + mask_other_perspective

            kernel = np.ones((8,8),np.uint8)
            erosion = cv2.erode(mask_perspective, kernel, iterations = 1) # experimental
            dilation = cv2.dilate(mask_perspective, kernel, iterations = 1) # experimental
            ret,thresh = cv2.threshold(mask_perspective, 40, 255, 0)
            im2,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            if len(contours) != 0 :
                c = max(contours, key = cv2.contourArea)
                cv2.drawContours(flip2, [c], -1, (0, 255, 0), -1)
                cv2.drawContours(flip2, [c], -1, (0, 0, 255), 1)
                M = cv2.moments(c)

                try :
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    center = (cx,cy)
                    extTop = tuple(c[c[:, :, 1].argmin()][0])
                    extBot = tuple(c[c[:, :, 1].argmax()][0])
                    extLeft = tuple(c[c[:, :, 0].argmin()][0])
                    extRight = tuple(c[c[:, :, 0].argmax()][0])
                    cv2.circle(flip2, (cx,cy), 5, (0, 0, 255), -1)
                    cv2.circle(flip2, extLeft, 5, (0, 0, 255), -1) 
                    cv2.circle(flip2, extRight, 5, (0, 0, 255), -1)

                except ZeroDivisionError :
                    print("No Road Detected")
            break

        # Image Processing X - X #  
        while data_yield <= 11  :
            # mask perspective
            mask_gray_perspective = cv2.inRange(hsv_perspective, low_gray_X, up_gray_X)
            mask_white_perspective = cv2.inRange(hsv_perspective, low_white_X, up_white_X)
            mask_other_perspective = cv2.inRange(hsv_perspective, low_other_X, up_other_X)
            mask_perspective= mask_gray_perspective + mask_white_perspective + mask_other_perspective

            kernel = np.ones((8,8),np.uint8)
            erosion = cv2.erode(mask_perspective, kernel, iterations = 1) # experimental
            dilation = cv2.dilate(mask_perspective, kernel, iterations = 1) # experimental
            ret,thresh = cv2.threshold(mask_perspective, 40, 255, 0)
            im2,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            if len(contours) != 0 :
                c = max(contours, key = cv2.contourArea)
                cv2.drawContours(flip, [c], -1, (0, 255, 0), -1)
                cv2.drawContours(flip, [c], -1, (0, 0, 255), 1)
                M = cv2.moments(c)

                try :
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    center = (cx,cy)
                    extTop = tuple(c[c[:, :, 1].argmin()][0])
                    extBot = tuple(c[c[:, :, 1].argmax()][0])
                    extLeft = tuple(c[c[:, :, 0].argmin()][0])
                    extRight = tuple(c[c[:, :, 0].argmax()][0])
                    cv2.circle(flip, (cx,cy), 5, (0, 0, 255), -1)
                    cv2.circle(flip, extLeft, 5, (0, 0, 255), -1) 
                    cv2.circle(flip, extRight, 5, (0, 0, 255), -1)

                except ZeroDivisionError :
                    print("No Road Detected")
            break

        # Image Processing XX - XX #  
        while 12 <= data_yield <= 20 :
            mask_gray_perspective = cv2.inRange(hsv_perspective, low_gray_XX, up_gray_XX)
            mask_white_perspective = cv2.inRange(hsv_perspective, low_white_XX, up_white_XX)
            mask_other_perspective = cv2.inRange(hsv_perspective, low_other_XX, up_other_XX)
            mask_other2_perspective = cv2.inRange(hsv_perspective, low_other2_XX, up_other2_XX)
            mask_perspective= mask_gray_perspective + mask_white_perspective + mask_other_perspective + mask_other2_perspective

            kernel = np.ones((8,8),np.uint8)
            erosion = cv2.erode(mask_perspective, kernel, iterations = 1) # experimental
            dilation = cv2.dilate(mask_perspective, kernel, iterations = 1) # experimental
            ret,thresh = cv2.threshold(mask_perspective, 40, 255, 0)
            im2,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            if len(contours) != 0 :
                c = max(contours, key = cv2.contourArea)
                cv2.drawContours(flip, [c], -1, (0, 255, 0), -1)
                cv2.drawContours(flip, [c], -1, (0, 0, 255), 1)
                M = cv2.moments(c)

                try :
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    center = (cx,cy)
                    extTop = tuple(c[c[:, :, 1].argmin()][0])
                    extBot = tuple(c[c[:, :, 1].argmax()][0])
                    extLeft = tuple(c[c[:, :, 0].argmin()][0])
                    extRight = tuple(c[c[:, :, 0].argmax()][0])
                    cv2.circle(flip, (cx,cy), 5, (0, 0, 255), -1)
                    cv2.circle(flip, extLeft, 5, (0, 0, 255), -1) 
                    cv2.circle(flip, extRight, 5, (0, 0, 255), -1)

                except ZeroDivisionError :
                    print("No Road Detected")
            break

        # Image Processing XXX - XXX #  
        while 21 <= data_yield <= 25 :
            mask_gray_perspective = cv2.inRange(hsv_perspective, low_gray_XXX, up_gray_XXX)
            mask_white_perspective = cv2.inRange(hsv_perspective, low_white_XXX, up_white_XXX)
            mask_other_perspective = cv2.inRange(hsv_perspective, low_other_XXX, up_other_XXX)
            mask_perspective= mask_gray_perspective + mask_white_perspective + mask_other_perspective

            kernel = np.ones((8,8),np.uint8)
            erosion = cv2.erode(mask_perspective, kernel, iterations = 1) # experimental
            dilation = cv2.dilate(mask_perspective, kernel, iterations = 1) # experimental
            ret,thresh = cv2.threshold(mask_perspective, 40, 255, 0)
            im2,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            if len(contours) != 0 :
                c = max(contours, key = cv2.contourArea)
                cv2.drawContours(flip, [c], -1, (0, 255, 0), -1)
                cv2.drawContours(flip, [c], -1, (0, 0, 255), 1)
                M = cv2.moments(c)

                try :
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    center = (cx,cy)
                    extTop = tuple(c[c[:, :, 1].argmin()][0])
                    extBot = tuple(c[c[:, :, 1].argmax()][0])
                    extLeft = tuple(c[c[:, :, 0].argmin()][0])
                    extRight = tuple(c[c[:, :, 0].argmax()][0])
                    cv2.circle(flip, (cx,cy), 5, (0, 0, 255), -1)
                    cv2.circle(flip, extLeft, 5, (0, 0, 255), -1) 
                    cv2.circle(flip, extRight, 5, (0, 0, 255), -1)

                except ZeroDivisionError :
                    print("No Road Detected")
            break

        # Image Processing XXXX - XXXX #  
        while 26 <= data_yield <= 30 :
            mask_gray_perspective = cv2.inRange(hsv_perspective, low_gray_XXXX, up_gray_XXXX)
            mask_white_perspective = cv2.inRange(hsv_perspective, low_white_XXXX, up_white_XXXX)
            mask_other_perspective = cv2.inRange(hsv_perspective, low_other_XXXX, up_other_XXXX)
            mask_other2_perspective = cv2.inRange(hsv_perspective, low_other2_XXXX, up_other2_XXXX)
            mask_perspective= mask_gray_perspective + mask_white_perspective + mask_other_perspective + mask_other2_perspective

            kernel = np.ones((8,8),np.uint8)
            erosion = cv2.erode(mask_perspective, kernel, iterations = 1) # experimental
            dilation = cv2.dilate(mask_perspective, kernel, iterations = 1) # experimental
            ret,thresh = cv2.threshold(mask_perspective, 40, 255, 0)
            im2,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            if len(contours) != 0 :
                c = max(contours, key = cv2.contourArea)
                cv2.drawContours(flip, [c], -1, (0, 255, 0), -1)
                cv2.drawContours(flip, [c], -1, (0, 0, 255), 1)
                M = cv2.moments(c)

                try :
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    center = (cx,cy)
                    extTop = tuple(c[c[:, :, 1].argmin()][0])
                    extBot = tuple(c[c[:, :, 1].argmax()][0])
                    extLeft = tuple(c[c[:, :, 0].argmin()][0])
                    extRight = tuple(c[c[:, :, 0].argmax()][0])
                    cv2.circle(flip, (cx,cy), 5, (0, 0, 255), -1)
                    cv2.circle(flip, extLeft, 5, (0, 0, 255), -1) 
                    cv2.circle(flip, extRight, 5, (0, 0, 255), -1)

                except ZeroDivisionError :
                    print("No Road Detected")
            break

        # Image Processing XXXXX - XXXXX #  
        while 31 <= data_yield <= 43 :
            mask_gray_perspective = cv2.inRange(hsv_perspective, low_gray_XXXXX, up_gray_XXXXX)
            mask_white_perspective = cv2.inRange(hsv_perspective, low_white_XXXXX, up_white_XXXXX)
            mask_other_perspective = cv2.inRange(hsv_perspective, low_other_XXXXX, up_other_XXXXX)
            mask_other2_perspective = cv2.inRange(hsv_perspective, low_other2_XXXXX, up_other2_XXXXX)
            mask_perspective= mask_gray_perspective + mask_white_perspective + mask_other_perspective + mask_other2_perspective

            kernel = np.ones((8,8),np.uint8)
            erosion = cv2.erode(mask_perspective, kernel, iterations = 1) # experimental
            dilation = cv2.dilate(mask_perspective, kernel, iterations = 1) # experimental
            ret,thresh = cv2.threshold(mask_perspective, 40, 255, 0)
            im2,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            if len(contours) != 0 :
                c = max(contours, key = cv2.contourArea)
                cv2.drawContours(flip, [c], -1, (0, 255, 0), -1)
                cv2.drawContours(flip, [c], -1, (0, 0, 255), 1)
                M = cv2.moments(c)

                try :
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    center = (cx,cy)
                    extTop = tuple(c[c[:, :, 1].argmin()][0])
                    extBot = tuple(c[c[:, :, 1].argmax()][0])
                    extLeft = tuple(c[c[:, :, 0].argmin()][0])
                    extRight = tuple(c[c[:, :, 0].argmax()][0])
                    cv2.circle(flip, (cx,cy), 5, (0, 0, 255), -1)
                    cv2.circle(flip, extLeft, 5, (0, 0, 255), -1) 
                    cv2.circle(flip, extRight, 5, (0, 0, 255), -1)

                except ZeroDivisionError :
                    print("No Road Detected")
            break

        # Image Processing XXXXXX - XXXXXX #  
        while 44 <= data_yield <= 53 :
            mask_gray_perspective = cv2.inRange(hsv_perspective, low_gray_XXXXXX, up_gray_XXXXXX)
            mask_white_perspective = cv2.inRange(hsv_perspective, low_white_XXXXXX, up_white_XXXXXX)
            mask_other_perspective = cv2.inRange(hsv_perspective, low_other_XXXXXX, up_other_XXXXXX)
            mask_perspective= mask_gray_perspective + mask_white_perspective + mask_other_perspective

            kernel = np.ones((8,8),np.uint8)
            erosion = cv2.erode(mask_perspective, kernel, iterations = 1) # experimental
            dilation = cv2.dilate(mask_perspective, kernel, iterations = 1) # experimental
            ret,thresh = cv2.threshold(mask_perspective, 40, 255, 0)
            im2,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            if len(contours) != 0 :
                c = max(contours, key = cv2.contourArea)
                cv2.drawContours(flip, [c], -1, (0, 255, 0), -1)
                cv2.drawContours(flip, [c], -1, (0, 0, 255), 1)
                M = cv2.moments(c)

                try :
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    center = (cx,cy)
                    extTop = tuple(c[c[:, :, 1].argmin()][0])
                    extBot = tuple(c[c[:, :, 1].argmax()][0])
                    extLeft = tuple(c[c[:, :, 0].argmin()][0])
                    extRight = tuple(c[c[:, :, 0].argmax()][0])
                    cv2.circle(flip, (cx,cy), 5, (0, 0, 255), -1)
                    cv2.circle(flip, extLeft, 5, (0, 0, 255), -1) 
                    cv2.circle(flip, extRight, 5, (0, 0, 255), -1)

                except ZeroDivisionError :
                    print("No Road Detected")
            break

        # Image Processing XXXXY - XXXXY #  
        while data_yield >= 127 :
            mask_gray_perspective = cv2.inRange(hsv_perspective, low_gray_XXXXY, up_gray_XXXXY)
            mask_white_perspective = cv2.inRange(hsv_perspective, low_white_XXXXY, up_white_XXXXY)
            mask_other_perspective = cv2.inRange(hsv_perspective, low_other_XXXXY, up_other_XXXXY)
            mask_perspective= mask_gray_perspective + mask_white_perspective + mask_other_perspective

            kernel = np.ones((8,8),np.uint8)
            erosion = cv2.erode(mask_perspective, kernel, iterations = 1) # experimental
            dilation = cv2.dilate(mask_perspective, kernel, iterations = 1) # experimental
            ret,thresh = cv2.threshold(mask_perspective, 40, 255, 0)
            im2,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            if len(contours) != 0 :
                c = max(contours, key = cv2.contourArea)
                cv2.drawContours(flip, [c], -1, (0, 255, 0), -1)
                cv2.drawContours(flip, [c], -1, (0, 0, 255), 1)
                M = cv2.moments(c)

                try :
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    center = (cx,cy)
                    extTop = tuple(c[c[:, :, 1].argmin()][0])
                    extBot = tuple(c[c[:, :, 1].argmax()][0])
                    extLeft = tuple(c[c[:, :, 0].argmin()][0])
                    extRight = tuple(c[c[:, :, 0].argmax()][0])
                    cv2.circle(flip, (cx,cy), 5, (0, 0, 255), -1)
                    cv2.circle(flip, extLeft, 5, (0, 0, 255), -1) 
                    cv2.circle(flip, extRight, 5, (0, 0, 255), -1)

                except ZeroDivisionError :
                    print("No Road Detected")
            break

        # Image Processing X7 - X7 #  
        while 54 <= data_yield <= 62 :
            mask_gray_perspective = cv2.inRange(hsv_perspective, low_gray_X7, up_gray_X7)
            mask_white_perspective = cv2.inRange(hsv_perspective, low_white_X7, up_white_X7)
            mask_other_perspective = cv2.inRange(hsv_perspective, low_other_X7, up_other_X7)
            mask_other2_perspective = cv2.inRange(hsv_perspective, low_other2_X7, up_other2_X7)
            mask_perspective= mask_gray_perspective + mask_white_perspective + mask_other_perspective + mask_other2_perspective

            kernel = np.ones((8,8),np.uint8)
            erosion = cv2.erode(mask_perspective, kernel, iterations = 1) # experimental
            dilation = cv2.dilate(mask_perspective, kernel, iterations = 1) # experimental
            ret,thresh = cv2.threshold(mask_perspective, 40, 255, 0)
            im2,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            if len(contours) != 0 :
                c = max(contours, key = cv2.contourArea)
                cv2.drawContours(flip, [c], -1, (0, 255, 0), -1)
                cv2.drawContours(flip, [c], -1, (0, 0, 255), 1)
                M = cv2.moments(c)

                try :
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    center = (cx,cy)
                    extTop = tuple(c[c[:, :, 1].argmin()][0])
                    extBot = tuple(c[c[:, :, 1].argmax()][0])
                    extLeft = tuple(c[c[:, :, 0].argmin()][0])
                    extRight = tuple(c[c[:, :, 0].argmax()][0])
                    cv2.circle(flip, (cx,cy), 5, (0, 0, 255), -1)
                    cv2.circle(flip, extLeft, 5, (0, 0, 255), -1) 
                    cv2.circle(flip, extRight, 5, (0, 0, 255), -1)

                except ZeroDivisionError :
                    print("No Road Detected")
            break

        # Image Processing X8 - X8 #  
        while 63 <= data_yield <= 97 :
            mask_gray_perspective = cv2.inRange(hsv_perspective, low_gray_X8, up_gray_X8)
            mask_white_perspective = cv2.inRange(hsv_perspective, low_white_X8, up_white_X8)
            mask_other_perspective = cv2.inRange(hsv_perspective, low_other_X8, up_other_X8)
            mask_perspective= mask_gray_perspective + mask_white_perspective + mask_other_perspective

            kernel = np.ones((8,8),np.uint8)
            erosion = cv2.erode(mask_perspective, kernel, iterations = 1) # experimental
            dilation = cv2.dilate(mask_perspective, kernel, iterations = 1) # experimental
            ret,thresh = cv2.threshold(mask_perspective, 40, 255, 0)
            im2,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            if len(contours) != 0 :
                c = max(contours, key = cv2.contourArea)
                cv2.drawContours(flip, [c], -1, (0, 255, 0), -1)
                cv2.drawContours(flip, [c], -1, (0, 0, 255), 1)
                M = cv2.moments(c)

                try :
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    center = (cx,cy)
                    extTop = tuple(c[c[:, :, 1].argmin()][0])
                    extBot = tuple(c[c[:, :, 1].argmax()][0])
                    extLeft = tuple(c[c[:, :, 0].argmin()][0])
                    extRight = tuple(c[c[:, :, 0].argmax()][0])
                    cv2.circle(flip, (cx,cy), 5, (0, 0, 255), -1)
                    cv2.circle(flip, extLeft, 5, (0, 0, 255), -1) 
                    cv2.circle(flip, extRight, 5, (0, 0, 255), -1)

                except ZeroDivisionError :
                    print("No Road Detected")
            break

        # Image Processing X9 - X9 #  
        while 98 <= data_yield <= 126 :
            mask_gray_perspective = cv2.inRange(hsv_perspective, low_gray_X9, up_gray_X9)
            mask_white_perspective = cv2.inRange(hsv_perspective, low_white_X9, up_white_X9)
            mask_other_perspective = cv2.inRange(hsv_perspective, low_other_X9, up_other_X9)
            mask_perspective= mask_gray_perspective + mask_white_perspective + mask_other_perspective

            kernel = np.ones((8,8),np.uint8)
            erosion = cv2.erode(mask_perspective, kernel, iterations = 1) # experimental
            dilation = cv2.dilate(mask_perspective, kernel, iterations = 1) # experimental
            ret,thresh = cv2.threshold(mask_perspective, 40, 255, 0)
            im2,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            if len(contours) != 0 :
                c = max(contours, key = cv2.contourArea)
                cv2.drawContours(flip, [c], -1, (0, 255, 0), -1)
                cv2.drawContours(flip, [c], -1, (0, 0, 255), 1)
                M = cv2.moments(c)

                try :
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    center = (cx,cy)
                    extTop = tuple(c[c[:, :, 1].argmin()][0])
                    extBot = tuple(c[c[:, :, 1].argmax()][0])
                    extLeft = tuple(c[c[:, :, 0].argmin()][0])
                    extRight = tuple(c[c[:, :, 0].argmax()][0])
                    cv2.circle(flip, (cx,cy), 5, (0, 0, 255), -1)
                    cv2.circle(flip, extLeft, 5, (0, 0, 255), -1) 
                    cv2.circle(flip, extRight, 5, (0, 0, 255), -1)

                except ZeroDivisionError :
                    print("No Road Detected")
            break

        # INPUT TEXT
        cv2.putText(resize,'Original', (10,40), font, 1, (0,0,255), 2, cv2.LINE_AA)
        cv2.putText(frame, 'ADAPTIVE', (10,40), font, 1, (0,0,255), 2, cv2.LINE_AA)
        cv2.putText(frame2, 'NOT-ADAPTIVE', (10,40), font, 1, (0,0,255), 2, cv2.LINE_AA)
        cv2.putText(flip_original, 'Original', (10,40), font, 1, (0,0,255), 2, cv2.LINE_AA)
        cv2.putText(flip2, 'NOT-ADAPTIVE', (10,40), font, 1, (0,0,255), 2, cv2.LINE_AA)
        cv2.putText(flip, 'ADAPTIVE', (10,40), font, 1, (0,0,255), 2, cv2.LINE_AA)

        # SHOW RESULT
        horizontal_show = np.hstack((frame2, frame, resize))
        horizontal_show_perspective = np.hstack((flip2, flip, flip_original))
        horizontal_blurdanori = np.hstack((resize_calibrated, frame3))
        horizontal_perspective = np.hstack((frame3, flip3 ))
        cv2.imshow('before and after', horizontal_perspective)
        cv2.imshow("masknormal", mask_perspective)
        cv2.imshow('original dan blur', horizontal_blurdanori)
        cv2.imshow('Perspective NOT-ADAPTIVE', flip2)
        cv2.imshow('NOT-ADAPTIVE', frame2)

        # SAVE RESULT
        result_path = "E:\\Grady\\Programming\\Python\\TugasAkhirGrady\\Result Gabungan"
        cv2.imshow("show",mask_perspective)
        cv2.imwrite('E:\\Grady\\Programming\\Python\\TugasAkhirGrady\\Result Gabungan\\{:>0}.jpg'.format(i), mask_perspective)
        i += 1


        cv2.waitKey(1)
        key = cv2.waitKey(25)
        if key == 27:
            break

# Start
thread1 = threading.Thread(target = image_processing)
thread1.start()

