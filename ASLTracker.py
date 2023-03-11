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


while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

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
            
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            print(prediction,index)

        else:
            k = imgSize/w
            hCal = math.ceil(k*h)
            imgResize = cv2.resize(imgCrop,(imgSize,hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize-hCal)/2)
            imgWhite[hGap:hCal+hGap, :] = imgResize #putting white as backgound
            
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        
        cv2.putText(imgOutput, labels[index],(x,y-20),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255),2)
        #cv2.rectangle(imgOutput,(x-offset,y-offset), (x+w+offset,y+h+offset), (255,0,255),4) rectangle around hand

        cv2.imshow("ImageCrop",imgCrop)
        cv2.imshow("ImageWhite",imgWhite)


        cv2.putText(img, "Jeremy Orr ©", (100, 100), cv2.FONT_HERSHEY_COMPLEX,2,(0,0,0),2)


    cv2.imshow("Image",imgOutput) #on the image output now chnage to img for other thing
    cv2.waitKey(1)



