# Blurs and anonymizes faces by detecting faces using deep neural networks
import numpy as np
import cv2
import time
import os
import imutils
from imutils.video import VideoStream

def anonymize_face_simple(image, factor=3.0):
    (h,w) = image.shape[:2]
    kW = int(w/factor)
    kH = int(h/factor)

    # Since GaussianBlur function doesn't take even kernel sizes, we have to conver them to odd.
    if kW%2 == 0:
        kW -= 1
    
    if kH%2 == 0:
        kH -= 1
    #Check gaussian blur kernel size 
    return cv2.GaussianBlur(image, (kW,kH),0)

def anonymize_face_pixelate(image, blocks=10):
    (h,w) = image.shape[:2]
    # numpy.linspace returns evenly spaced numbers in a 
    # range. In this case, it will divide the image horizontally
    # and vertically into 3 blocks.
    xSteps = np.linspace(0,w,blocks+1, dtype="int")
    ySteps = np.linspace(0,h,blocks+1,dtype="int")

    for i in range(1, len(ySteps)):
        for j in range(1,len(xSteps)):
            
            startX = xSteps[j-1]
            startY = ySteps[i-1]
            endX = xSteps[j]
            endY = ySteps[i]
            
            roi = image[startY: endY, startX: endX]
            (B,G,R) = [int(x) for x in cv2.mean(roi)[:3]]
            # You can hardcode the (B,G,R) values to blur the 
            # image with a color of your choice.
            cv2.rectangle(image,(startX,startY),(endX,endY),(B,G,R),-1)

    return image

base_path = os.path.dirname(os.path.abspath(__file__))

protoxt_path = os.path.join(base_path+'/model_data/deploy.prototxt')
caffemodel_path = os.path.join(base_path+'/model_data/weights.caffemodel')

model = cv2.dnn.readNetFromCaffe(protoxt_path,caffemodel_path)

print("Starting Video stream")
vs = VideoStream(src=0).start()
time.sleep(2.0)

while True:
    frame = vs.read()
    frame = imutils.resize(frame,width=400)

    (h,w) = frame.shape[:2]
    # (104.0,177.0,123.0) is the mean value for the ImageNet training set.
    blob = cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)),1.0,(h,w),(104.0,177.0,123.0)) 
    model.setInput(blob)
    detections = model.forward()

    for i in range(0,detections.shape[2]):
        box = detections[0,0,i,3:7]*np.array([w,h,w,h])
        (startX, startY, endX, endY) = box.astype("int")
        face = frame[startY:endY,startX:endX]
        confidence = detections[0,0,i,2]
        if(confidence>0.5):
            #Uncomment below for simple blur
            #face = anonymize_face_simple(face)
            face = anonymize_face_pixelate(face)
            frame[startY:endY,startX:endX] = face

    cv2.imshow("Frame",frame)

    key = cv2.waitKey(1)&0xFF

    if key==ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
vs.stream.release()
print("[INFO]Video stream stopped")