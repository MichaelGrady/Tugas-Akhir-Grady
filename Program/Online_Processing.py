# Created by : Michael Grady
# Mechanical Engineering AtmaJaya Catholic University of Indonesia
# For bachelor thesis : “The Development of Road Area Detection System Using Computer Vision Technique”
# Online Processing


import serial
import time
import cv2
import threading
import datetime
import glob
from openpyxl import load_workbook
import numpy as np
import matplotlib.pyplot as plt
import os
import math

######## arduino ########
ser = serial.Serial('COM7', 9600)
ser.flushInput()
def LDR_Data():
        ser_bytes = ser.readline()
        arduinoInt = int(ser_bytes)
        return arduinoInt
##########################

def live_processing():
    # DIST MTX KAMERA LOGITECH HD C270.
    dist = np.float64([[-0.131304517,2.53233311, 0.00159093065, -0.00267024626,-19.7256640]])

    mtx = np.float64([   [1880.88010, 0.            ,  621.613745],
                     [0.          , 1897.97466  ,  287.104560],
                     [0.          , 0.            ,  1.          ]    ])
    cap = cv2.VideoCapture(1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    i = 1

    while True :
        ret, frame = cap.read()
        arduinoData = LDR_Data()  
        print(arduinoData)
        # undistortion
        h,  w = frame.shape[:2]
        newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
        dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)
        x,y,w,h = roi
        dst = dst[y:y+h, x:x+w]
        resize = cv2.resize(dst, (450, 450), interpolation = cv2.INTER_LINEAR)
        frame_proc = cv2.GaussianBlur(resize, (5, 5), 0) # adaptive
        frame_proc2 = cv2.GaussianBlur(resize, (5, 5), 0) # fix
        hsv = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2HSV) # adaptive
        hsv2 = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2HSV) # fix
        
        pts1= np.float32([[0, 450],[450, 450],[30, 225],[420,225]])
        pts2 = np.float32([[0, 0], [450, 0], [0, 450], [450, 450]])
        cv2.circle(frame_proc, (pts1[0,0],pts1[0,1]), 5, (0, 255, 255), -1) 
        cv2.circle(frame_proc, (pts1[1,0],pts1[1,1]), 5, (0, 255, 255), -1) 
        cv2.circle(frame_proc, (pts1[2,0],pts1[2,1]), 5, (0, 255, 255), -1)
        cv2.circle(frame_proc, (pts1[3,0],pts1[3,1]), 5, (0, 255, 255), -1)
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        result = cv2.warpPerspective(frame_proc, matrix, (450,450))
        flipResult = cv2.flip(result, -1)
        flip_original = cv2.flip( flipResult, 1 ) # original
        flip = cv2.flip( flipResult, 1 ) # adaptive
        flip2 = cv2.flip( flipResult, 1 ) # fix

        hsv_perspective = cv2.cvtColor(flip, cv2.COLOR_BGR2HSV)
        hsv_perspective2 = cv2.cvtColor(flip, cv2.COLOR_BGR2HSV)

        ########## MASKING STORAGE ###########
        # HSV FIXED
        low_gray = np.array([22, -12, 104])
        up_gray = np.array([62, 28, 204])
        low_white = np.array([46, -12, 108])
        up_white = np.array([86,  28, 208])
        low_shadow = np.array([67, 5, 54])
        up_shadow = np.array([107, 45, 154])

        # LDR X - X # < 11
        low_gray_X = np.array([22, -11,  87])
        up_gray_X = np.array([62,  29, 187])
        low_white_X = np.array([63, -13,  92])
        up_white_X = np.array([103,  27, 192])
        low_shadow_X = np.array([40, 235, -43])
        up_shadow_X = np.array([80, 275,  57])

        # LDR XX - XX # 12-20
        low_gray_XX = np.array([22, -12, 104])
        up_gray_XX = np.array([62, 28, 204])
        low_white_XX = np.array([46, -12, 108])
        up_white_XX = np.array([86,  28, 208])
        low_other_XX = np.array([-30, -30,  17])
        up_other_XX = np.array([30,  30, 137])
        low_shadow_XX = np.array([67, 5, 54])
        up_shadow_XX = np.array([107, 45, 154])
        
        # LDR XXX - XXX # 21-25
        low_gray_XXX = np.array([23, -7, 87])
        up_gray_XXX = np.array([63,  33, 187])
        low_white_XXX = np.array([46, -9, 70])
        up_white_XXX = np.array([86,  31, 170])
        low_shadow_XXX = np.array([63, -13, 106])
        up_shadow_XXX = np.array([103, 27, 206])

         # LDR XXXX - XXXX # 26-30
        low_gray_XXXX = np.array([23, -8, 105])
        up_gray_XXXX = np.array([63, 32, 205])
        low_white_XXXX = np.array([50, -20, 100])
        up_white_XXXX = np.array([110, 40, 190])
        low_shadow_XXXX = np.array([63, -13, 106])
        up_shadow_XXXX = np.array([103, 27, 206])
        low_other_XXXX = np.array([40, -1, -8])
        up_other_XXXX = np.array([100,  59, 112])

         # LDR XXXXX - XXXXX # 31-43
        low_gray_XXXXX = np.array([20, -23,  43])
        up_gray_XXXXX = np.array([80,  37, 163])
        low_white_XXXXX = np.array([36,  -1, -16])
        up_white_XXXXX = np.array([96,  59, 104])
        low_other_XXXXX = np.array([79, -8, 23])
        up_other_XXXXX = np.array([146,  76, 143])
        low_shadow_XXXXX = np.array([53, -18, 44])
        up_shadow_XXXXX = np.array([123,  42, 164])

         # LDR XXXXXX - XXXXXX # 44-53
        low_gray_XXXXXX = np.array([12, 0, -22])
        up_gray_XXXXXX = np.array([96, 64, 103])
        low_white_XXXXXX = np.array([36,  -1, -16])
        up_white_XXXXXX = np.array([96,  59, 104])
        low_shadow_XXXXXX = np.array([73, -18, 93])
        up_shadow_XXXXXX = np.array([133,  42, 213])

        # LDR X7 - X7 # 54-62
        low_gray_X7 = np.array([86, -2, 79])
        up_gray_X7 = np.array([146,  58, 199])
        low_white_X7 = np.array([50, -20, 100])
        up_white_X7 = np.array([110, 40, 190])
        low_shadow_X7 = np.array([63, -13, 106])
        up_shadow_X7 = np.array([103, 27, 206])
        low_other_X7 = np.array([53,  -7, -15])
        up_other_X7 = np.array([113,  53, 105])

        # LDR X8 - X8 # 63-97
        low_gray_X8 = np.array([53, -23,  78])
        up_gray_X8 = np.array([113, 37, 198])
        low_white_X8 = np.array([81, -13,  91])
        up_white_X8 = np.array([141,  47, 211])
        low_shadow_X8 = np.array([80, 39, 87])
        up_shadow_X8 = np.array([140, 99, 207])

        # LDR X9 - X9 # 98-126
        low_gray_X9 = np.array([82, -10, 127])
        up_gray_X9 = np.array([142, 50, 247])
        low_white_X9 = np.array([73, -19, 106])
        up_white_X9 = np.array([133,  41, 226])
        low_shadow_X9 = np.array([80, 18, 36])
        up_shadow_X9 = np.array([140, 78, 156])

         # LDR XXXXY - XXXXY # > 452
        low_gray_XXXXY = np.array([96, 59, 63])
        up_gray_XXXXY = np.array([136,  99, 163])
        low_white_XXXXY = np.array([96, 39, 45])
        up_white_XXXXY = np.array([136,  79, 145])
        low_shadow_XXXXY = np.array([85, 118,  -2])
        up_shadow_XXXXY = np.array([130, 190,  98])

        ############# WITHOUT PERSPECTIVE TRANSFORM ################
        # Image Processing LDR FIXED HSV FIXED #
        while arduinoData >= 0  :
            # masking berdasarkan warna
            mask_gray = cv2.inRange(hsv2, low_gray, up_gray)
            mask_white = cv2.inRange(hsv2, low_white, up_white)
            mask_shadow = cv2.inRange(hsv2, low_shadow, up_shadow)
            mask = mask_gray + mask_white + mask_shadow
            mask_all = cv2.rectangle(mask,(450,0),(0,175),(0,255,0),-1)

            kernel = np.ones((8,8),np.uint8)
            erosion = cv2.erode(mask_all, kernel, iterations = 1) # experimental
            dilation = cv2.dilate(mask_all, kernel, iterations = 1) # experimental
            ret,thresh = cv2.threshold(mask_all, 40, 255, 0)
            im2,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            if len(contours) != 0 :
                c = max(contours, key = cv2.contourArea)
                cv2.drawContours(frame_proc2, [c], -1, (0, 255, 0), -1)
                cv2.drawContours(frame_proc2, [c], -1, (0, 0, 255), 1)
                M = cv2.moments(c)

                try :
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    center = (cx,cy)
                    extTop = tuple(c[c[:, :, 1].argmin()][0])
                    extBot = tuple(c[c[:, :, 1].argmax()][0])
                    extLeft = tuple(c[c[:, :, 0].argmin()][0])
                    extRight = tuple(c[c[:, :, 0].argmax()][0])
                    # cv2.circle(frame_proc2, (cx,cy), 5, (0, 0, 255), -1)
                    # cv2.circle(frame_proc2, extLeft, 5, (0, 0, 255), -1) 
                    # cv2.circle(frame_proc2, extRight, 5, (0, 0, 255), -1)

                except ZeroDivisionError :
                    print("No Road Detected")
            break

        # Image Processing X - X #
        while arduinoData <= 11 :
            # masking berdasarkan warna
            mask_gray = cv2.inRange(hsv, low_gray_X, up_gray_X)
            mask_white = cv2.inRange(hsv, low_white_X, up_white_X)
            mask_shadow = cv2.inRange(hsv, low_shadow_X, up_shadow_X)
            mask = mask_gray + mask_white + mask_shadow
            mask_all = cv2.rectangle(mask,(450,0),(0,175),(0,255,0),-1)

            kernel = np.ones((8,8),np.uint8)
            erosion = cv2.erode(mask_all, kernel, iterations = 1) # experimental
            dilation = cv2.dilate(mask_all, kernel, iterations = 1) # experimental
            ret,thresh = cv2.threshold(mask_all, 40, 255, 0)
            im2,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            if len(contours) != 0 :
                c = max(contours, key = cv2.contourArea)
                cv2.drawContours(frame_proc, [c], -1, (0, 255, 0), -1)
                cv2.drawContours(frame_proc, [c], -1, (0, 0, 255), 1)
                M = cv2.moments(c)
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                center = (cx,cy)
                extTop = tuple(c[c[:, :, 1].argmin()][0])
                extBot = tuple(c[c[:, :, 1].argmax()][0])
                extLeft = tuple(c[c[:, :, 0].argmin()][0])
                extRight = tuple(c[c[:, :, 0].argmax()][0])
                # cv2.circle(frame_proc, (cx,cy), 5, (0, 0, 255), -1)
                # cv2.circle(frame_proc, extLeft, 5, (0, 0, 255), -1) 
                # cv2.circle(frame_proc, extRight, 5, (0, 0, 255), -1)

            break

        # Image Processing XX - XX #  
        while 12 <= arduinoData <= 20 :
            # masking berdasarkan warna
            mask_gray = cv2.inRange(hsv, low_gray_XX, up_gray_XX)
            mask_white = cv2.inRange(hsv, low_white_XX, up_white_XX)
            mask_other = cv2.inRange(hsv, low_other_XX, up_other_XX)
            mask_shadow = cv2.inRange(hsv, low_shadow_XX, up_shadow_XX)
            mask = mask_gray + mask_white + mask_shadow
            mask_all = cv2.rectangle(mask,(450,0),(0,175),(0,255,0),-1)

            kernel = np.ones((8,8),np.uint8)
            erosion = cv2.erode(mask_all, kernel, iterations = 1) # experimental
            dilation = cv2.dilate(mask_all, kernel, iterations = 1) # experimental
            ret,thresh = cv2.threshold(mask_all, 40, 255, 0)
            im2,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            if len(contours) != 0 :
                c = max(contours, key = cv2.contourArea)
                cv2.drawContours(frame_proc, [c], -1, (0, 255, 0), -1)
                cv2.drawContours(frame_proc, [c], -1, (0, 0, 255), 1)
                M = cv2.moments(c)

                try :
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    center = (cx,cy)
                    extTop = tuple(c[c[:, :, 1].argmin()][0])
                    extBot = tuple(c[c[:, :, 1].argmax()][0])
                    extLeft = tuple(c[c[:, :, 0].argmin()][0])
                    extRight = tuple(c[c[:, :, 0].argmax()][0])
                    # cv2.circle(frame_proc, (cx,cy), 5, (0, 0, 255), -1)
                    # cv2.circle(frame_proc, extLeft, 5, (0, 0, 255), -1) 
                    # cv2.circle(frame_proc, extRight, 5, (0, 0, 255), -1)

                except ZeroDivisionError :
                    print("No Road Detected")

            break

        # Image Processing XXX - XXX #  
        while 21 <= arduinoData <= 25 :
            # masking berdasarkan warna
            mask_gray = cv2.inRange(hsv, low_gray_XXX, up_gray_XXX)
            mask_white = cv2.inRange(hsv, low_white_XXX, up_white_XXX)
            mask_shadow = cv2.inRange(hsv, low_shadow_XXX, up_shadow_XXX)
            mask = mask_gray + mask_white + mask_shadow
            mask_all = cv2.rectangle(mask,(450,0),(0,175),(0,255,0),-1)

            kernel = np.ones((8,8),np.uint8)
            erosion = cv2.erode(mask_all, kernel, iterations = 1) # experimental
            dilation = cv2.dilate(mask_all, kernel, iterations = 1) # experimental
            ret,thresh = cv2.threshold(mask_all, 40, 255, 0)
            im2,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            if len(contours) != 0 :
                c = max(contours, key = cv2.contourArea)
                cv2.drawContours(frame_proc, [c], -1, (0, 255, 0), -1)
                cv2.drawContours(frame_proc, [c], -1, (0, 0, 255), 1)
                M = cv2.moments(c)
                try :
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    center = (cx,cy)
                    extTop = tuple(c[c[:, :, 1].argmin()][0])
                    extBot = tuple(c[c[:, :, 1].argmax()][0])
                    extLeft = tuple(c[c[:, :, 0].argmin()][0])
                    extRight = tuple(c[c[:, :, 0].argmax()][0])
                    # cv2.circle(frame_proc, (cx,cy), 5, (0, 0, 255), -1)
                    # cv2.circle(frame_proc, extLeft, 5, (0, 0, 255), -1) 
                    # cv2.circle(frame_proc, extRight, 5, (0, 0, 255), -1)

                except ZeroDivisionError :
                    print("No Road Detected")

            break

        # Image Processing XXXX - XXXX #  
        while 26 <= arduinoData <= 30 :
            # masking berdasarkan warna
            mask_gray = cv2.inRange(hsv, low_gray_XXXX, up_gray_XXXX)
            mask_white = cv2.inRange(hsv, low_white_XXXX, up_white_XXXX)
            mask_shadow = cv2.inRange(hsv, low_shadow_XXXX, up_shadow_XXXX)
            mask_other = cv2.inRange(hsv, low_other_XXXX, up_other_XXXX)
            mask = mask_gray + mask_white + mask_shadow + mask_other
            mask_all = cv2.rectangle(mask,(450,0),(0,175),(0,255,0),-1)

            kernel = np.ones((8,8),np.uint8)
            erosion = cv2.erode(mask_all, kernel, iterations = 1) # experimental
            dilation = cv2.dilate(mask_all, kernel, iterations = 1) # experimental
            ret,thresh = cv2.threshold(mask_all, 40, 255, 0)
            im2,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            if len(contours) != 0 :
                c = max(contours, key = cv2.contourArea)
                cv2.drawContours(frame_proc, [c], -1, (0, 255, 0), -1)
                cv2.drawContours(frame_proc, [c], -1, (0, 0, 255), 1)
                M = cv2.moments(c)
                try :
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    center = (cx,cy)
                    extTop = tuple(c[c[:, :, 1].argmin()][0])
                    extBot = tuple(c[c[:, :, 1].argmax()][0])
                    extLeft = tuple(c[c[:, :, 0].argmin()][0])
                    extRight = tuple(c[c[:, :, 0].argmax()][0])
                    # cv2.circle(frame_proc, (cx,cy), 5, (0, 0, 255), -1)
                    # cv2.circle(frame_proc, extLeft, 5, (0, 0, 255), -1) 
                    # cv2.circle(frame_proc, extRight, 5, (0, 0, 255), -1)

                except ZeroDivisionError :
                    print("No Road Detected")

            break

        # Image Processing XXXXX - XXXXX #  
        while 31 <= arduinoData <= 43 :
            # masking berdasarkan warna
            mask_gray = cv2.inRange(hsv, low_gray_XXXXX, up_gray_XXXXX)
            mask_white = cv2.inRange(hsv, low_white_XXXXX, up_white_XXXXX)
            mask_shadow = cv2.inRange(hsv, low_shadow_XXXXX, up_shadow_XXXXX)
            mask_other = cv2.inRange(hsv, low_other_XXXXX, up_other_XXXXX)
            mask = mask_gray + mask_white + mask_shadow + mask_other
            mask_all = cv2.rectangle(mask,(450,0),(0,175),(0,255,0),-1)

            kernel = np.ones((8,8),np.uint8)
            erosion = cv2.erode(mask_all, kernel, iterations = 1) # experimental
            dilation = cv2.dilate(mask_all, kernel, iterations = 1) # experimental
            ret,thresh = cv2.threshold(mask_all, 40, 255, 0)
            im2,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            if len(contours) != 0 :
                c = max(contours, key = cv2.contourArea)
                cv2.drawContours(frame_proc, [c], -1, (0, 255, 0), -1)
                cv2.drawContours(frame_proc, [c], -1, (0, 0, 255), 1)
                M = cv2.moments(c)
                try :
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    center = (cx,cy)
                    extTop = tuple(c[c[:, :, 1].argmin()][0])
                    extBot = tuple(c[c[:, :, 1].argmax()][0])
                    extLeft = tuple(c[c[:, :, 0].argmin()][0])
                    extRight = tuple(c[c[:, :, 0].argmax()][0])
                    # cv2.circle(frame_proc, (cx,cy), 5, (0, 0, 255), -1)
                    # cv2.circle(frame_proc, extLeft, 5, (0, 0, 255), -1) 
                    # cv2.circle(frame_proc, extRight, 5, (0, 0, 255), -1)

                except ZeroDivisionError :
                    print("No Road Detected")

            break

        # Image Processing XXXXY - XXXXY #  
        while arduinoData >= 452 :
            # masking berdasarkan warna
            mask_gray = cv2.inRange(hsv, low_gray_XXXXY, up_gray_XXXXY)
            mask_white = cv2.inRange(hsv, low_white_XXXXY, up_white_XXXXY)
            mask_shadow = cv2.inRange(hsv, low_shadow_XXXXY, up_shadow_XXXXY)
            mask = mask_gray + mask_white + mask_shadow
            mask_all = cv2.rectangle(mask,(450,0),(0,175),(0,255,0),-1)

            kernel = np.ones((8,8),np.uint8)
            erosion = cv2.erode(mask_all, kernel, iterations = 1) # experimental
            dilation = cv2.dilate(mask_all, kernel, iterations = 1) # experimental
            ret,thresh = cv2.threshold(mask_all, 40, 255, 0)
            im2,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            if len(contours) != 0 :
                c = max(contours, key = cv2.contourArea)
                cv2.drawContours(frame_proc, [c], -1, (0, 255, 0), -1)
                cv2.drawContours(frame_proc, [c], -1, (0, 0, 255), 1)
                M = cv2.moments(c)
                try :
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    center = (cx,cy)
                    extTop = tuple(c[c[:, :, 1].argmin()][0])
                    extBot = tuple(c[c[:, :, 1].argmax()][0])
                    extLeft = tuple(c[c[:, :, 0].argmin()][0])
                    extRight = tuple(c[c[:, :, 0].argmax()][0])
                    # cv2.circle(frame_proc, (cx,cy), 5, (0, 0, 255), -1)
                    # cv2.circle(frame_proc, extLeft, 5, (0, 0, 255), -1) 
                    # cv2.circle(frame_proc, extRight, 5, (0, 0, 255), -1)

                except ZeroDivisionError :
                    print("No Road Detected")

            break

        # Image Processing XXXXXX - XXXXXX #  
        while 44 <= arduinoData <= 53 :
            # masking berdasarkan warna
            mask_gray = cv2.inRange(hsv, low_gray_XXXXXX, up_gray_XXXXXX)
            mask_white = cv2.inRange(hsv, low_white_XXXXXX, up_white_XXXXXX)
            mask_shadow = cv2.inRange(hsv, low_shadow_XXXXXX, up_shadow_XXXXXX)
            mask = mask_gray + mask_white + mask_shadow
            mask_all = cv2.rectangle(mask,(450,0),(0,175),(0,255,0),-1)

            kernel = np.ones((8,8),np.uint8)
            erosion = cv2.erode(mask_all, kernel, iterations = 1) # experimental
            dilation = cv2.dilate(mask_all, kernel, iterations = 1) # experimental
            ret,thresh = cv2.threshold(mask_all, 40, 255, 0)
            im2,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            if len(contours) != 0 :
                c = max(contours, key = cv2.contourArea)
                cv2.drawContours(frame_proc, [c], -1, (0, 255, 0), -1)
                cv2.drawContours(frame_proc, [c], -1, (0, 0, 255), 1)
                M = cv2.moments(c)
                try :
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    center = (cx,cy)
                    extTop = tuple(c[c[:, :, 1].argmin()][0])
                    extBot = tuple(c[c[:, :, 1].argmax()][0])
                    extLeft = tuple(c[c[:, :, 0].argmin()][0])
                    extRight = tuple(c[c[:, :, 0].argmax()][0])
                    # cv2.circle(frame_proc, (cx,cy), 5, (0, 0, 255), -1)
                    # cv2.circle(frame_proc, extLeft, 5, (0, 0, 255), -1) 
                    # cv2.circle(frame_proc, extRight, 5, (0, 0, 255), -1)

                except ZeroDivisionError :
                    print("No Road Detected")

            break 

        # Image Processing X7 - X7 #  
        while 54 <= arduinoData <= 62 :
            # masking berdasarkan warna
            mask_gray = cv2.inRange(hsv, low_gray_X7, up_gray_X7)
            mask_white = cv2.inRange(hsv, low_white_X7, up_white_X7)
            mask_shadow = cv2.inRange(hsv, low_shadow_X7, up_shadow_X7)
            mask_other = cv2.inRange(hsv, low_other_X7, up_other_X7)
            mask = mask_gray + mask_white + mask_shadow + mask_other
            mask_all = cv2.rectangle(mask,(450,0),(0,175),(0,255,0),-1)

            kernel = np.ones((8,8),np.uint8)
            erosion = cv2.erode(mask_all, kernel, iterations = 1) # experimental
            dilation = cv2.dilate(mask_all, kernel, iterations = 1) # experimental
            ret,thresh = cv2.threshold(mask_all, 40, 255, 0)
            im2,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            if len(contours) != 0 :
                c = max(contours, key = cv2.contourArea)
                cv2.drawContours(frame_proc, [c], -1, (0, 255, 0), -1)
                cv2.drawContours(frame_proc, [c], -1, (0, 0, 255), 1)
                M = cv2.moments(c)
                try :
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    center = (cx,cy)
                    extTop = tuple(c[c[:, :, 1].argmin()][0])
                    extBot = tuple(c[c[:, :, 1].argmax()][0])
                    extLeft = tuple(c[c[:, :, 0].argmin()][0])
                    extRight = tuple(c[c[:, :, 0].argmax()][0])
                    # cv2.circle(frame_proc, (cx,cy), 5, (0, 0, 255), -1)
                    # cv2.circle(frame_proc, extLeft, 5, (0, 0, 255), -1) 
                    # cv2.circle(frame_proc, extRight, 5, (0, 0, 255), -1)

                except ZeroDivisionError :
                    print("No Road Detected")

            break                

        # Image Processing X8 - X8 #  
        while 63 <= arduinoData <= 97 :
            # masking berdasarkan warna
            mask_gray = cv2.inRange(hsv, low_gray_X8, up_gray_X8)
            mask_white = cv2.inRange(hsv, low_white_X8, up_white_X8)
            mask_shadow = cv2.inRange(hsv, low_shadow_X8, up_shadow_X8)
            mask = mask_gray + mask_white + mask_shadow
            mask_all = cv2.rectangle(mask,(450,0),(0,175),(0,255,0),-1)

            kernel = np.ones((8,8),np.uint8)
            erosion = cv2.erode(mask_all, kernel, iterations = 1) # experimental
            dilation = cv2.dilate(mask_all, kernel, iterations = 1) # experimental
            ret,thresh = cv2.threshold(mask_all, 40, 255, 0)
            im2,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            if len(contours) != 0 :
                c = max(contours, key = cv2.contourArea)
                cv2.drawContours(frame_proc, [c], -1, (0, 255, 0), -1)
                cv2.drawContours(frame_proc, [c], -1, (0, 0, 255), 1)
                M = cv2.moments(c)
                try :
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    center = (cx,cy)
                    extTop = tuple(c[c[:, :, 1].argmin()][0])
                    extBot = tuple(c[c[:, :, 1].argmax()][0])
                    extLeft = tuple(c[c[:, :, 0].argmin()][0])
                    extRight = tuple(c[c[:, :, 0].argmax()][0])
                    # cv2.circle(frame_proc, (cx,cy), 5, (0, 0, 255), -1)
                    # cv2.circle(frame_proc, extLeft, 5, (0, 0, 255), -1) 
                    # cv2.circle(frame_proc, extRight, 5, (0, 0, 255), -1)

                except ZeroDivisionError :
                    print("No Road Detected")

            break 

        # Image Processing X9 - X9 #  
        while 98 <= arduinoData <= 126 :
            # masking berdasarkan warna
            mask_gray = cv2.inRange(hsv, low_gray_X9, up_gray_X9)
            mask_white = cv2.inRange(hsv, low_white_X9, up_white_X9)
            mask_shadow = cv2.inRange(hsv, low_shadow_X9, up_shadow_X9)
            mask = mask_gray + mask_white + mask_shadow
            mask_all = cv2.rectangle(mask,(450,0),(0,175),(0,255,0),-1)

            kernel = np.ones((8,8),np.uint8)
            erosion = cv2.erode(mask_all, kernel, iterations = 1) # experimental
            dilation = cv2.dilate(mask_all, kernel, iterations = 1) # experimental
            ret,thresh = cv2.threshold(mask_all, 40, 255, 0)
            im2,contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            if len(contours) != 0 :
                c = max(contours, key = cv2.contourArea)
                cv2.drawContours(frame_proc, [c], -1, (0, 255, 0), -1)
                cv2.drawContours(frame_proc, [c], -1, (0, 0, 255), 1)
                M = cv2.moments(c)
                try :
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    center = (cx,cy)
                    extTop = tuple(c[c[:, :, 1].argmin()][0])
                    extBot = tuple(c[c[:, :, 1].argmax()][0])
                    extLeft = tuple(c[c[:, :, 0].argmin()][0])
                    extRight = tuple(c[c[:, :, 0].argmax()][0])
                    # cv2.circle(frame_proc, (cx,cy), 5, (0, 0, 255), -1)
                    # cv2.circle(frame_proc, extLeft, 5, (0, 0, 255), -1) 
                    # cv2.circle(frame_proc, extRight, 5, (0, 0, 255), -1)

                except ZeroDivisionError :
                    print("No Road Detected")

            break 

        ########## WITH PERSPECTIVE TRANSFORM ##########
        # Image Processing LDR FIXED HSV FIXED #
        while arduinoData >= 0  :
            # mask perspective
            mask_gray_perspective = cv2.inRange(hsv_perspective2, low_gray, up_gray)
            mask_white_perspective = cv2.inRange(hsv_perspective2, low_white, up_white)
            mask_shadow_perspective = cv2.inRange(hsv_perspective2, low_shadow, up_shadow)
            mask_perspective= mask_gray_perspective + mask_white_perspective + mask_shadow_perspective

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
                    # cv2.circle(flip2, (cx,cy), 5, (0, 0, 255), -1)
                    # cv2.circle(flip2, extLeft, 5, (0, 0, 255), -1) 
                    # cv2.circle(flip2, extRight, 5, (0, 0, 255), -1)

                except ZeroDivisionError :
                    print("No Road Detected")
            break

        # Image Processing X - X #  
        while arduinoData <= 11  :
            # mask perspective
            mask_gray_perspective = cv2.inRange(hsv_perspective, low_gray_X, up_gray_X)
            mask_white_perspective = cv2.inRange(hsv_perspective, low_white_X, up_white_X)
            mask_shadow_perspective = cv2.inRange(hsv_perspective, low_shadow_X, up_shadow_X)
            mask_perspective= mask_gray_perspective + mask_white_perspective + mask_shadow_perspective

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
                    # cv2.circle(flip, (cx,cy), 5, (0, 0, 255), -1)
                    # cv2.circle(flip, extLeft, 5, (0, 0, 255), -1) 
                    # cv2.circle(flip, extRight, 5, (0, 0, 255), -1)

                except ZeroDivisionError :
                    print("No Road Detected")
            break

        # Image Processing XX - XX #  
        while 12 <= arduinoData <= 20 :
            mask_gray_perspective = cv2.inRange(hsv_perspective, low_gray_XX, up_gray_XX)
            mask_white_perspective = cv2.inRange(hsv_perspective, low_white_XX, up_white_XX)
            mask_shadow_perspective = cv2.inRange(hsv_perspective, low_shadow_XX, up_shadow_XX)
            mask_other_perspective = cv2.inRange(hsv_perspective, low_other_XX, up_other_XX)
            mask_perspective= mask_gray_perspective + mask_white_perspective + mask_shadow_perspective + mask_other_perspective

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
                    # cv2.circle(flip, (cx,cy), 5, (0, 0, 255), -1)
                    # cv2.circle(flip, extLeft, 5, (0, 0, 255), -1) 
                    # cv2.circle(flip, extRight, 5, (0, 0, 255), -1)

                except ZeroDivisionError :
                    print("No Road Detected")
            break

        # Image Processing XXX - XXX #  
        while 21 <= arduinoData <= 25 :
            mask_gray_perspective = cv2.inRange(hsv_perspective, low_gray_XXX, up_gray_XXX)
            mask_white_perspective = cv2.inRange(hsv_perspective, low_white_XXX, up_white_XXX)
            mask_shadow_perspective = cv2.inRange(hsv_perspective, low_shadow_XXX, up_shadow_XXX)
            mask_perspective= mask_gray_perspective + mask_white_perspective + mask_shadow_perspective

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
                    # cv2.circle(flip, (cx,cy), 5, (0, 0, 255), -1)
                    # cv2.circle(flip, extLeft, 5, (0, 0, 255), -1) 
                    # cv2.circle(flip, extRight, 5, (0, 0, 255), -1)

                except ZeroDivisionError :
                    print("No Road Detected")
            break

        # Image Processing XXXX - XXXX #  
        while 26 <= arduinoData <= 30 :
            mask_gray_perspective = cv2.inRange(hsv_perspective, low_gray_XXXX, up_gray_XXXX)
            mask_white_perspective = cv2.inRange(hsv_perspective, low_white_XXXX, up_white_XXXX)
            mask_shadow_perspective = cv2.inRange(hsv_perspective, low_shadow_XXXX, up_shadow_XXXX)
            mask_other_perspective = cv2.inRange(hsv_perspective, low_other_XXXX, up_other_XXXX)
            mask_perspective= mask_gray_perspective + mask_white_perspective + mask_shadow_perspective + mask_other_perspective

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
                    # cv2.circle(flip, (cx,cy), 5, (0, 0, 255), -1)
                    # cv2.circle(flip, extLeft, 5, (0, 0, 255), -1) 
                    # cv2.circle(flip, extRight, 5, (0, 0, 255), -1)

                except ZeroDivisionError :
                    print("No Road Detected")
            break

        # Image Processing XXXXX - XXXXX #  
        while 31 <= arduinoData <= 43 :
            mask_gray_perspective = cv2.inRange(hsv_perspective, low_gray_XXXXX, up_gray_XXXXX)
            mask_white_perspective = cv2.inRange(hsv_perspective, low_white_XXXXX, up_white_XXXXX)
            mask_shadow_perspective = cv2.inRange(hsv_perspective, low_shadow_XXXXX, up_shadow_XXXXX)
            mask_other_perspective = cv2.inRange(hsv_perspective, low_other_XXXXX, up_other_XXXXX)
            mask_perspective= mask_gray_perspective + mask_white_perspective + mask_shadow_perspective + mask_other_perspective

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
                    # cv2.circle(flip, (cx,cy), 5, (0, 0, 255), -1)
                    # cv2.circle(flip, extLeft, 5, (0, 0, 255), -1) 
                    # cv2.circle(flip, extRight, 5, (0, 0, 255), -1)

                except ZeroDivisionError :
                    print("No Road Detected")
            break

        # Image Processing XXXXXX - XXXXXX #  
        while 44 <= arduinoData <= 53 :
            mask_gray_perspective = cv2.inRange(hsv_perspective, low_gray_XXXXXX, up_gray_XXXXXX)
            mask_white_perspective = cv2.inRange(hsv_perspective, low_white_XXXXXX, up_white_XXXXXX)
            mask_shadow_perspective = cv2.inRange(hsv_perspective, low_shadow_XXXXXX, up_shadow_XXXXXX)
            mask_perspective= mask_gray_perspective + mask_white_perspective + mask_shadow_perspective

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
                    # cv2.circle(flip, (cx,cy), 5, (0, 0, 255), -1)
                    # cv2.circle(flip, extLeft, 5, (0, 0, 255), -1) 
                    # cv2.circle(flip, extRight, 5, (0, 0, 255), -1)

                except ZeroDivisionError :
                    print("No Road Detected")
            break

        # Image Processing XXXXY - XXXXY #  
        while arduinoData >= 452 :
            mask_gray_perspective = cv2.inRange(hsv_perspective, low_gray_XXXXY, up_gray_XXXXY)
            mask_white_perspective = cv2.inRange(hsv_perspective, low_white_XXXXY, up_white_XXXXY)
            mask_shadow_perspective = cv2.inRange(hsv_perspective, low_shadow_XXXXY, up_shadow_XXXXY)
            mask_perspective= mask_gray_perspective + mask_white_perspective + mask_shadow_perspective

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
                    # cv2.circle(flip, (cx,cy), 5, (0, 0, 255), -1)
                    # cv2.circle(flip, extLeft, 5, (0, 0, 255), -1) 
                    # cv2.circle(flip, extRight, 5, (0, 0, 255), -1)

                except ZeroDivisionError :
                    print("No Road Detected")
            break

        # Image Processing X7 - X7 #  
        while 54 <= arduinoData <= 62 :
            mask_gray_perspective = cv2.inRange(hsv_perspective, low_gray_X7, up_gray_X7)
            mask_white_perspective = cv2.inRange(hsv_perspective, low_white_X7, up_white_X7)
            mask_shadow_perspective = cv2.inRange(hsv_perspective, low_shadow_X7, up_shadow_X7)
            mask_other_perspective = cv2.inRange(hsv_perspective, low_other_X7, up_other_X7)
            mask_perspective= mask_gray_perspective + mask_white_perspective + mask_shadow_perspective + mask_other_perspective

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
                    # cv2.circle(flip, (cx,cy), 5, (0, 0, 255), -1)
                    # cv2.circle(flip, extLeft, 5, (0, 0, 255), -1) 
                    # cv2.circle(flip, extRight, 5, (0, 0, 255), -1)

                except ZeroDivisionError :
                    print("No Road Detected")
            break

        # Image Processing X8 - X8 #  
        while 63 <= arduinoData <= 97 :
            mask_gray_perspective = cv2.inRange(hsv_perspective, low_gray_X8, up_gray_X8)
            mask_white_perspective = cv2.inRange(hsv_perspective, low_white_X8, up_white_X8)
            mask_shadow_perspective = cv2.inRange(hsv_perspective, low_shadow_X8, up_shadow_X8)
            mask_perspective= mask_gray_perspective + mask_white_perspective + mask_shadow_perspective

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
                    # cv2.circle(flip, (cx,cy), 5, (0, 0, 255), -1)
                    # cv2.circle(flip, extLeft, 5, (0, 0, 255), -1) 
                    # cv2.circle(flip, extRight, 5, (0, 0, 255), -1)

                except ZeroDivisionError :
                    print("No Road Detected")
            break

        # Image Processing X9 - X9 #  
        while 98 <= arduinoData <= 126 :
            mask_gray_perspective = cv2.inRange(hsv_perspective, low_gray_X9, up_gray_X9)
            mask_white_perspective = cv2.inRange(hsv_perspective, low_white_X9, up_white_X9)
            mask_shadow_perspective = cv2.inRange(hsv_perspective, low_shadow_X9, up_shadow_X9)
            mask_perspective= mask_gray_perspective + mask_white_perspective + mask_shadow_perspective

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
                    # cv2.circle(flip, (cx,cy), 5, (0, 0, 255), -1)
                    # cv2.circle(flip, extLeft, 5, (0, 0, 255), -1) 
                    # cv2.circle(flip, extRight, 5, (0, 0, 255), -1)

                except ZeroDivisionError :
                    print("No Road Detected")
            break

        # INPUT TEXT
        cv2.putText(resize,'Original', (10,40), font, 1, (0,0,255), 2, cv2.LINE_AA)
        cv2.putText(frame_proc, 'ADAPTIVE', (10,40), font, 1, (0,0,255), 2, cv2.LINE_AA)
        cv2.putText(frame_proc2, 'NOT-ADAPTIVE', (10,40), font, 1, (0,0,255), 2, cv2.LINE_AA)
        cv2.putText(flip_original, 'Original', (10,40), font, 1, (0,0,255), 2, cv2.LINE_AA)
        cv2.putText(flip2, 'NOT-ADAPTIVE', (10,40), font, 1, (0,0,255), 2, cv2.LINE_AA)
        cv2.putText(flip, 'ADAPTIVE', (10,40), font, 1, (0,0,255), 2, cv2.LINE_AA)

        horizontal_show = np.hstack((frame_proc, frame_proc2, resize))
        horizontal_show_perspective = np.hstack((flip, flip2, flip_original))
        # cv2.imshow("NOT-ADAPTIVE, ADAPTIVE, Original",horizontal_show)	
        cv2.imshow('PERSPECTIVE NOT-ADAPTIVE, ADAPTIVE, AND ORIGINAL', horizontal_show_perspective)
        cv2.imwrite('E:\\Grady\\Programming\\Python\\TugasAkhirGrady\\Online_Adaptive\\{:>0}.jpg'.format(i), flip)
        cv2.imwrite('E:\\Grady\\Programming\\Python\\TugasAkhirGrady\\Online_NonAdaptive\\{:>0}.jpg'.format(i), flip2)
        cv2.imwrite('E:\\Grady\\Programming\\Python\\TugasAkhirGrady\\Online_Original\\{:>0}.jpg'.format(i), flip_original)
        cv2.imwrite('E:\\Grady\\Programming\\Python\\TugasAkhirGrady\\OnlineProc\\{:>0}.jpg'.format(i), resize)    

        i += 1

        off = cv2.waitKey(5) & 0xFF
        if off == 27:
            break

    cv2.destroyAllWindows()
    cap.release()

thread1 = threading.Thread(target = live_processing)
thread1.start()
