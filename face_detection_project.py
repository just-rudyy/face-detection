from os import read
import cv2 as cv

vid = cv.VideoCapture(0)
harr_casc = cv.CascadeClassifier('haar_casc.xml')

while(True):
    ret,frame = vid.read()
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    face_react = harr_casc.detectMultiScale(gray, 1.1, 7)
    for (x,y,w,h) in face_react:
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv.putText(frame, 'Face detected',((x+w)//2,(y+h)//2),cv.FONT_HERSHEY_TRIPLEX,1.0,(0,255,0),2)
    cv.imshow('face detection',frame)
    if cv.waitKey(1) & 0xFF == ord('q'): 
        break





