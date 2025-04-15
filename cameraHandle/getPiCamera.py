from picamera.array import PiRGBArray
from picamera import PiCamera
import imutils
import time
import cv2

camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640,480))

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

time.sleep(0.1)
def capture_loop():
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        image = frame.array
        face_cascade = cv2.CascadeClassifier('../assets/haarcascade_frontalface_alt.xml')
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,1.1,5)
        print("found "+str(len(faces))+"faces")
        
        for (x,y,w,h) in faces:
            cv2.rectangle(image, (x,y),(x+w,y+h),(255,255,0),2)
        
        cv2.imshow("OK",image)
        key = cv2.waitKey(1) & 0xFF
        
        rawCapture.truncate(0)
        if key== ord("q"):
            break
        
capture_loop()