import cv2
import numpy as np
from sklearn.cluster import KMeans

from ultralytics import YOLO
from ultralytics import SAM

# Load the YOLO model
model = YOLO("yolo11n.pt")
model_seg = SAM("sam2.1_b.pt")

# Open the video file
cap = cv2.VideoCapture(0)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        results = model(frame, classes=[61], conf=0.25)
        for result in results:
            xyxy = result.boxes.xyxy   
            if len(xyxy) > 0:
                x1, y1, x2, y2 = map(float, xyxy[0].tolist())
                bounding_box = [x1, y1, x2, y2]
            else:
                bounding_box = None
            print(xyxy) 

        annotated_frame = results[0].plot()
        # tensor([[ 762.5332,  370.4412, 1111.9058, 1052.6259]])
        print(bounding_box)
        
        esults_seg = model_seg(annotated_frame, bboxes=bounding_box,  stream=True)
        for mask in esults_seg:
            masks = mask.plot()
        cv2.imshow("YOLO Inference", masks)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()

