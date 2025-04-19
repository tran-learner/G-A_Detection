from picamera2 import Picamera2

import time
import cv2
import os
import sys
sys.path.append(os.path.abspath("../G&A_DETECTION/"))
from utils.ga_model_handle import predict_ga

def capture_loop():
    face_cascade = cv2.CascadeClassifier('assets/haarcascade_frontalface_alt.xml')
    
    frame_count = 0
    process_every_n_frames =30
    
    camera = PiCamera2()
    config = camera.create_preview_configuration(main={"format":"RGB888", "size":(700,500)})
    camera.configure(config)
    camera.start()
    time.sleep(1)


    while (True):
        frame = camera.capture_array()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7, minSize=(30, 30))
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y),(x+w,y+h),(255,255,0),3)
            if frame_count % process_every_n_frames ==0:
                face_img = frame[y:y+h, x:x+w].copy()
                age, gender = predict_ga(face_img)
                # print(a)
                
            label = f"{gender}, {age}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2)
        frame_count +=1
        if (frame_count>325):
            frame_count=0
        
        label = f"{gender}, {age}"        
        
        cv2.imshow("camera",frame)
        if cv2.waitKey(1)==ord("q"):
            break
    cv2.destroyAllWindows()
    camera.stop()

