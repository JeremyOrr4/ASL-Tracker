#DATA COLLECTION PART

import cv2;
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1) #max hands
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")


offset = 20 #Create offset for smaller box
imgSize = 300

folder="Data/C"
counter = 0

labels = ["A","B","C"]

#for printing the input text
letterList = []
printedString = ""

#Quit Key
def quit_key_pressed():
    # Wait for a key press
    key = cv2.waitKey(1) & 0xFF

    # If the "q" key was pressed, return True
    if key == ord('q'):
        return True

    return False


def Create_Printed_String(char):
    global letterList, printedString

    letterList.append(labels[index])    
        
    if letterList.count(char) > 25:
        printedString = printedString + char
        letterList = []

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if quit_key_pressed():
            break
    
    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox'] #Create the smaller box for our hand

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255
        imgCrop = img[y-offset:y+h+offset,x-offset:x+w+offset]

        imgCropShape = imgCrop.shape

        aspectRatio = h/w
        if aspectRatio >1:
            k = imgSize/h
            wCal = math.ceil(k*w)
            imgResize = cv2.resize(imgCrop,(wCal,imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((300-wCal)/2)
            imgWhite[:, wGap:wCal+wGap] = imgResize #putting white as backgound
            
            prediction, index = classifier.getPrediction(imgWhite, draw=True)
            #print(prediction,index)

        else:
            k = imgSize/w
            hCal = math.ceil(k*h)
            imgResize = cv2.resize(imgCrop,(imgSize,hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize-hCal)/2)
            imgWhite[hGap:hCal+hGap, :] = imgResize #putting white as backgound
            
            prediction, index = classifier.getPrediction(imgWhite, draw=True)
        
        cv2.putText(imgOutput, labels[index],(x,y-20), cv2.FONT_HERSHEY_DUPLEX,2,(255,0,255),2)
        #cv2.rectangle(imgOutput,(x-offset,y-offset), (x+w+offset,y+h+offset), (255,0,255),4) rectangle around hand

        Create_Printed_String(labels[index])     

        cv2.imshow("ImageCrop",imgCrop)
        cv2.imshow("ImageWhite",imgWhite)

        #Exit key code
        if quit_key_pressed():
            break



    #Printing the string
    cv2.putText(imgOutput, printedString, (50, 50), cv2.FONT_HERSHEY_DUPLEX,0.9,(255,255,255),1) 
    
    #Drawing the creator code
    CreatorText = "Created by Jeremy Orr"
    #cv2.rectangle(imgOutput, (0,400), (50,460), (0, 255, 0), thickness=10)
    cv2.putText(imgOutput, CreatorText, (0, 475), cv2.FONT_HERSHEY_DUPLEX,0.9,(255,255,255),1) 
    
    cv2.imshow("Image",imgOutput) #Show new image without backend.
    cv2.waitKey(1)


