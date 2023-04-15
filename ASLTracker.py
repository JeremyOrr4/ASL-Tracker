import cv2;
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
from gtts import gTTS
from playsound import playsound
import os
import time


cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")


offset = 20
imgSize = 300

labels = ["A","B","C"]

#for printing the input text
global StringMaxLength, CyclesToRegisterInput

letterList = []
printedString = ""
appenedPrintedString = ""
StringMaxLength = 21
CyclesToRegisterInput = 1

n = 10
StringList = ['' for i in range(n)]
StringListIndex = 0

CreatorText = "ASL to Speech & Text      Jeremy Orr"

#User Functions
class UserFunctions():

    #Quit Key
    def quit_key_pressed(self):
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            return True

        return False


    def Create_Printed_String(self,char):
        global letterList, printedString, appenedPrintedString, StringList, StringListIndex

        if len(StringList[StringListIndex]) < StringMaxLength:
            letterList.append(labels[index])    

            if char == "A" and letterList.count("A") > CyclesToRegisterInput:
                self.DeleteChar()

            if char == "TextToSpeech" and letterList.count("TextToSpeech") > CyclesToRegisterInput:
                self.StringToSpeech()

            if char == "B" and letterList.count("B") > CyclesToRegisterInput:
                self.WriteToFile()

            if letterList.count(char) > CyclesToRegisterInput:
                StringList[StringListIndex] = StringList[StringListIndex] + char
                letterList = []

                if len(StringList[StringListIndex]) == StringMaxLength:
                    LastCharOfPrintedString = StringList[StringListIndex][-1]
                    StringListIndex = StringListIndex + 1
                    StringList[StringListIndex] = LastCharOfPrintedString

    def DeleteChar(self):
        global letterList, printedString, appenedPrintedString, StringList, StringListIndex

        if (StringListIndex >= 1 and len(StringList[StringListIndex]) == 0):
            StringListIndex = StringListIndex - 1

        StringList[StringListIndex] = StringList[StringListIndex][:-1]

        letterList = []

        
    def WriteToFile(self):
        global letterList, printedString, appenedPrintedString, StringList, StringListIndex, n

        with open("Sign_Language_Output_Text.txt", mode="wt") as textFile:
            FinalOutput = ''.join(StringList)

            textFile.write(FinalOutput)
            StringList = ['' for i in range(n)]
            letterList = []
            StringListIndex = 0

    def StringToSpeech(self):
        global letterList, printedString, appenedPrintedString, StringList, StringListIndex, n

        FinalOutput = ''.join(StringList)
        outputAsText = gTTS(FinalOutput, lang='en',slow = False)
        outputAsText.save("Sign_Language_Output_Speech.mp3")
        playsound('Sign_Language_Output_Speech.mp3')
        os.remove("Sign_Language_Output_Speech.mp3")
        StringList = ['' for i in range(n)]
        letterList = []
        StringListIndex = 0

User = UserFunctions()

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if User.quit_key_pressed():
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

            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize-hCal)/2)
            imgWhite[hGap:hCal+hGap, :] = imgResize #putting white as backgound
            
            prediction, index = classifier.getPrediction(imgWhite, draw=True)
        
        cv2.putText(imgOutput, labels[index],(x+26,y-20), cv2.FONT_HERSHEY_DUPLEX,2,(0,0,255),2)

        cv2.imshow("ImageCrop",imgCrop)
        cv2.imshow("ImageWhite",imgWhite)

        if (labels[index] != "Standby"):
            User.Create_Printed_String(labels[index])     

        if User.quit_key_pressed():
            break



    #Top section
    cv2.rectangle(imgOutput, (0,0), (750,40), (0,0,0), -1)

    cv2.putText(imgOutput, CreatorText, (30, 30), cv2.FONT_HERSHEY_DUPLEX,0.9,(255,255,255),1) 
    
    
    cv2.rectangle(imgOutput, (0,438), (750,600), (0,0,0), -1)


    cv2.putText(imgOutput, StringList[StringListIndex], (0, 475), cv2.FONT_HERSHEY_DUPLEX,1.5,(255,255,255),2)
    
    cv2.imshow("Image",imgOutput) #Show new image without backend.
    cv2.waitKey(1)


