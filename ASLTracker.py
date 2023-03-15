import cv2;
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
from gtts import gTTS
import os

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1) #max hands
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")


offset = 20 #Create offset for smaller box
imgSize = 300

labels = ["A","B","C"]

#for printing the input text
global StringMaxLength, CyclesToRegisterInput

letterList = []
printedString = ""
appenedPrintedString = ""
StringMaxLength = 22
CyclesToRegisterInput = 22

CreatorText = "ASL to Speech & Text      Jeremy Orr"

#Text to speech variables


#Quit Key
def quit_key_pressed():
    # Wait for a key press
    key = cv2.waitKey(1) & 0xFF

    # If the "q" key was pressed, return True
    if key == ord('q'):
        return True

    return False


def Create_Printed_String(char):
    global letterList, printedString, appenedPrintedString
    
    if len(printedString) < StringMaxLength:
        letterList.append(labels[index])    

        if char == "Delete" and letterList.count("Delete") > CyclesToRegisterInput:
            DeleteChar(True)

        if letterList.count(char) > CyclesToRegisterInput:
            printedString = printedString + char
            letterList = []

            if len(printedString) == StringMaxLength:
                LastCharOfPrintedString = printedString[-1]
                appenedPrintedString = LastCharOfPrintedString      
            

    else:
        letterList.append(labels[index])     

        if char == "Delete" and letterList.count("Delete") > CyclesToRegisterInput: 
            DeleteChar(False)

        if letterList.count(char) > CyclesToRegisterInput:
            appenedPrintedString = appenedPrintedString + char
            letterList = [] 


def DeleteChar(firstBlock):
    global letterList, printedString, appenedPrintedString
    
    if firstBlock == True:
        printedString = printedString[:-1]
        letterList = []
    
    elif firstBlock == False:
        if len(appenedPrintedString) == 0:
            DeleteChar(True)

        appenedPrintedString = appenedPrintedString[:-1]
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
        
        if aspectRatio > 1:
            k = imgSize/h
            wCal = math.ceil(k*w)

            #Fix off Screen Error
            if imgCrop.size > 0:
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            else:
                continue

            #imgResize = cv2.resize(imgCrop,(wCal,imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((300-wCal)/2)
            imgWhite[:, wGap:wCal+wGap] = imgResize #putting white as backgound
            
            prediction, index = classifier.getPrediction(imgWhite, draw=True)
            #print(prediction,index)

        else:
            k = imgSize/w
            hCal = math.ceil(k*h)
            
            #Fix off Screen Error
            if imgCrop.size > 0:
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            else:
                continue

            #imgResize = cv2.resize(imgCrop,(imgSize,hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize-hCal)/2)
            imgWhite[hGap:hCal+hGap, :] = imgResize #putting white as backgound
            
            prediction, index = classifier.getPrediction(imgWhite, draw=True)
        
        cv2.putText(imgOutput, labels[index],(x+26,y-20), cv2.FONT_HERSHEY_DUPLEX,2,(0,0,255),2)
        #cv2.rectangle(imgOutput,(x-offset,y-offset), (x+w+offset,y+h+offset), (255,0,255),4) rectangle around hand

        cv2.imshow("ImageCrop",imgCrop)
        cv2.imshow("ImageWhite",imgWhite)

        
        Create_Printed_String(labels[index])     

        if quit_key_pressed():
            break



    #Top section
    cv2.rectangle(imgOutput, (0,0), (750,40), (0,0,0), -1)

    cv2.putText(imgOutput, CreatorText, (30, 30), cv2.FONT_HERSHEY_DUPLEX,0.9,(255,255,255),1) 
    
    #Drawing output box
    #Createing the Box for the output text
    cv2.rectangle(imgOutput, (0,438), (750,600), (0,0,0), -1)

    if len(printedString) < StringMaxLength:
        cv2.putText(imgOutput, printedString, (0, 475), cv2.FONT_HERSHEY_DUPLEX,1.5,(255,255,255),2) 

    else:
        cv2.putText(imgOutput, appenedPrintedString, (0, 475), cv2.FONT_HERSHEY_DUPLEX,1.5,(255,255,255),2) 
    
    cv2.imshow("Image",imgOutput) #Show new image without backend.
    cv2.waitKey(1)


