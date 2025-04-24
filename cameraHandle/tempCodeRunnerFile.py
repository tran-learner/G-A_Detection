 faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7, minSize=(30, 30))
        current_labels = []
        for (x,y,w,h) in faces:
            padding_x = int(w*0.1)
            padding_y = int(h*0.2)
            x1 = max(x - padding_x, 0)
            y1 = max(y - int(padding_y*0.6), 0)
            x2 = min(x + w + padding_x, frame.shape[1])
            y2 = min(y + h + int(padding_y*0.6), frame.shape[0])
            # cv2.rectangle(frame, (x,y),(x+w,y+h),(255,255,0),3)
            cv2.rectangle(frame, (x1,y1),(x2,y2),(255,255,0),3)
            
            if frame_count % process_every_n_frames ==0:
                face_img = frame[y1:y2, x1:x2].copy()
                age, gender = lite_predict_ga(face_img)
                # print(a)
                
                label = f"{gender}, {age}"
                current_labels.append((x,y,label))