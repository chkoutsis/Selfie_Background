import cv2
import mediapipe as mp
import numpy as np

mp_selfie = mp.solutions.selfie_segmentation
cap = cv2.VideoCapture(0)

# create with statement for model 
with mp_selfie.SelfieSegmentation(model_selection=0) as model: 
    while cap.isOpened():
        ret, frame = cap.read()
        # apply segmentation  
        res = model.process(frame)
        mask = np.stack((res.segmentation_mask,)*3, axis=-1) > 0.8 
        
        #if i want to use blur as a background
        segmented_image = np.where(mask, frame, cv2.blur(frame, (40,40)))  
        
        #if i want to use background colour (not blur)
        background = np.zeros(frame.shape, np.uint8)
        background[:] = (255,0,0)   #colour
        
        #if i want to use image as a background
        path = '/the path of the background picture'
        img = cv2.imread(path)
        img= cv2.resize(img, (640,480))
        image_np = np.array(img)
        background[:]=image_np
             
        segmented_image = np.where(mask, frame, background)
        cv2.imshow('Background', segmented_image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
            
cap.release()
cv2.destroyAllWindows()