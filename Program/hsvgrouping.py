# Original Author : Alieldin Ayman
# HSV Plotter Modifier : Michael Grady
# For bachelor thesis : “The Development of Road Area Detection System Using Computer Vision Technique”

import cv2
import numpy as np
import sys
import tkinter as tk
from tkinter import filedialog

image_hsv = None
pixel = (0,0,0) 

ftypes = [
    ('JPG', '*.jpg;*.JPG;*.JPEG'), 
    ('PNG', '*.png;*.PNG'),
    ('GIF', '*.gif;*.GIF'),
]

def pick_color(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel = image_hsv[y,x]

        # HSV with tolerance
        upper =  np.array([pixel[0] + 30, pixel[1] + 30, pixel[2] + 50])
        lower =  np.array([pixel[0] - 30, pixel[1] - 30, pixel[2] - 50])
        print(lower, upper)

        image_mask = cv2.inRange(image_hsv,lower,upper)
        cv2.imshow("Mask",image_mask)

def main():
    global image_hsv, pixel
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes = ftypes)
    image_src = cv2.imread(file_path)
    cv2.imshow("BGR",image_src)

    #CREATE THE HSV FROM THE BGR IMAGE
    image_hsv = cv2.cvtColor(image_src,cv2.COLOR_BGR2HSV)
    cv2.imshow("HSV",image_hsv)

    #CALLBACK FUNCTION
    cv2.setMouseCallback("HSV", pick_color)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=='__main__':
    main()
