# Synchronized Data Recorder
# Created by : Michael Grady
# Mechanical Engineering AtmaJaya Catholic University of Indonesia
# For bachelor thesis : “The Development of Road Area Detection System Using Computer Vision Technique”

import serial
import time
import csv
import cv2
import threading
import datetime

def foo(nomorFrame):
    hitungFrame = nomorFrame + 1
    yield(hitungFrame)    

######## arduino ########
ser = serial.Serial('COM7', 9600)
ser.flushInput()
def LDR_Data():
        ser_bytes = ser.readline()
        arduinoInt = int(ser_bytes)
        return arduinoInt
##########################

def proses_LDR():
    prosesArduino = LDR_Data() 
    return prosesArduino

########## video ##########
def video_data():
    cap = cv2.VideoCapture(1)
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output = cv2.VideoWriter('Data Video.mp4',fourcc, 24.0, (640,480))
    nomorFrame = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        arduinoData = LDR_Data()        

        if ret == True:
            cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),(10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)    
            output.write(frame)
            nomorFrame = nomorFrame + 1
            print(nomorFrame, arduinoData)
            with open("Frame dan LDR.csv","a", newline='') as f:
                writer = csv.writer(f,delimiter=" ")
                writer.writerow([time.strftime("%H : %M : %S : LDR_DATA = "), nomorFrame, arduinoData])
            cv2.imshow('frame',frame) 
            #print(arduinoData)
        key = cv2.waitKey(25)
        if key == 27:
            break
    cap.release()
    output.release()
    cv2.destroyAllWindows()
#################################

thread1 = threading.Thread(target = video_data)
thread2 = threading.Thread(target = LDR_Data)
thread1.start()
thread2.start()
thread1.join()
thread2.join()

'''
def create_excel():
    while True :
        with open("Combined Data.csv","a", newline='') as f:
            writer = csv.writer(f,delimiter=" ")
            writer.writerow([])
        key = cv2.waitKey(25)
        if key == 27:
            break
'''

''' 
# LDR_DATA
ser = serial.Serial('COM3')
ser.flushInput()
 
while True:
    ser_bytes = ser.readline()
    decoded_bytes = int(ser_bytes[0:len(ser_bytes)-2].decode("utf-8"))
    print(decoded_bytes)

    with open("LDR_DATA.csv","a", newline='') as f:
        writer = csv.writer(f,delimiter=" ")
        writer.writerow([time.strftime("%H : %M : %S : LDR_DATA = "),decoded_bytes])
'''

'''
# Video Data

cap = cv2.VideoCapture(0)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('asdasd.mp4',fourcc, 24.0, (640,480))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:

        # write the flipped frame
        out.write(frame)

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
'''
