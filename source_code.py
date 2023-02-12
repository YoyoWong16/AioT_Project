
# Code Reference : https://www.geeksforgeeks.org/python-eye-blink-detection-project/amp/ 

#All the imports go here

import numpy as np

import cv2

import time
 
#Initializing the face and eye cascade classifiers from xml files

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
 
#Variable store execution state

first_read = True
 
#Starting the video capture

cap = cv2.VideoCapture(0)

ret,img = cap.read()
 

t=0

while(ret):

    ret,img = cap.read()

    #Converting the recorded image to grayscale

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Applying filter to remove impurities

    gray = cv2.bilateralFilter(gray,5,1,1)
 

    #Detecting the face for region of image to be fed to eye classifier

    faces = face_cascade.detectMultiScale(gray, 1.3, 5,minSize=(200,200))
    

    if(len(faces)>0):

        for (x,y,w,h) in faces:

            img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
 

            #roi_face is face which is input to eye classifier

            roi_face = gray[y:y+h,x:x+w]

            roi_face_clr = img[y:y+h,x:x+w]

            eyes = eye_cascade.detectMultiScale(roi_face,1.3,5,minSize=(50,50))
 

            #Examining the length of eyes object for eyes

            if(len(eyes)>=1):

                #Check if program is running for detection 
                    cv2.putText(img,

                   "eyes open",(100,100),

                    cv2.FONT_HERSHEY_PLAIN, 3, 

                    (0,255,0),2)
                    
                    t+=1

                    cv2.waitKey(100)
                        
                    if (t>30): #the value of time that open eyes, can be changed
                        
                        print('Please remeber blink your eyes')
                    
            else:

                    #This will print on console and restart the algorithm
                    cv2.putText(img,

                    "eyes closed",(100,100),

                    cv2.FONT_HERSHEY_PLAIN, 3, 

                    (0,255,0),2)
                    
                    cv2.waitKey(100)
                    
                    if (t>10): #the value of time that open eyes, for preventing error, can be changed
                    
                        print("your eyes open for", t/10, 's')
                    
                    t=0
                    
    else:

        cv2.putText(img,

        "No face detected",(100,100),

        cv2.FONT_HERSHEY_PLAIN, 3, 

        (0,255,0),2)
 

    #Controlling the algorithm with keys

    cv2.imshow('img',img)
 
cap.release()
cv2.destroyAllWindows()

