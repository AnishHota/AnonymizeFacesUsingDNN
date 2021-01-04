# Blurs and anonymizes the faces by detecting faces using Haar Cascade Classifier.
import numpy as np
import cv2
import time
import imutils
import os
from imutils.video import VideoStream

def anonymize_face_simple(image, factor=3.0):
#Increasing the factor will increase the blurriness
    (h,w) = image.shape[:2]
    kW = int(w/factor)
    kH = int(h/factor)

    if kW%2 == 0:
        kW -= 1

    if kH%2 == 0:
        kH -= 1

    return cv2.GaussianBlur(image, (kW,kH), 0)

def anonymize_face_pixelate(image, blocks=10):
    (h,w) = image.shape[:2]
    #numpy.linspace returns evenly spaced numbers in a 
    #range. In this case, it will divide the image horizontally
    # and vertically into blocks+1
    xSteps = np.linspace(0,w,blocks+1,dtype="int")
    ySteps = np.linspace(0,h,blocks+1,dtype="int")

    for i in range(1,len(ySteps)):
        for j in range(1,len(xSteps)):

            startX = xSteps[j-1]
            startY = ySteps[i-1]
            endX = xSteps[j]
            endY = ySteps[i]

            roi = image[startY:endY, startX: endX]
            (B,G,R) = [int(x) for x in cv2.mean(roi)[:3]]
            cv2.rectangle(image,(startX,startY), (endX,endY), (B,G,R),-1)

    return image

base_path = os.path.dirname(os.path.abspath(__file__))
face_cascade = cv2.CascadeClassifier(base_path+'/model_data/haarcascade_frontalface_default.xml')
print("Starting video stream")
vs = VideoStream(src=0).start()
time.sleep(2.0)

while True:
    frame = vs.read()
    frame = imutils.resize(frame,width=400)
    
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    
    for (x,y,w,h) in faces:
        face = frame[y:y+h, x:x+w]
        #Uncomment below for simple blur
        #face = anonymize_face_simple(face) 
        face = anonymize_face_pixelate(face)
        frame[y:y+h, x:x+w] = face
    
    cv2.imshow("Frame",frame)
    
    key = cv2.waitKey(1)&0xFF
    
    if key==ord("q"):
        break
cv2.destroyAllWindows()
vs.stop()
vs.stream.release()
print("Video stream stopped.")