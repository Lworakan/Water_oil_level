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
        
        esults_seg = model_seg(annotated_frame, bboxes=bounding_box, stream=True)
        for mask in esults_seg:
            masks = mask.plot()
            
            # Get the segmentation mask (binary mask)
            if hasattr(mask, 'masks') and mask.masks is not None:
                seg_mask = mask.masks.data[0].cpu().numpy()  # Get first mask
                
                # Resize mask to match frame size if needed
                if seg_mask.shape != frame.shape[:2]:
                    seg_mask = cv2.resize(seg_mask, (frame.shape[1], frame.shape[0]))
                
                # Convert to binary mask
                binary_mask = (seg_mask > 0.5).astype(np.uint8)
                
                # Extract pixels within the mask from original frame
                masked_pixels = frame[binary_mask == 1]
                
                if len(masked_pixels) > 0:
                    # Reshape for clustering (n_pixels, 3 color channels)
                    pixels_reshaped = masked_pixels.reshape(-1, 3)
                    
                    # Perform K-means clustering with 2 clusters
                    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(pixels_reshaped)
                    
                    # Calculate percentage of each cluster
                    unique, counts = np.unique(labels, return_counts=True)
                    total_pixels = len(labels)
                    
                    percentages = {}
                    for cluster_id, count in zip(unique, counts):
                        percentage = (count / total_pixels) * 100
                        percentages[f'Cluster {cluster_id}'] = percentage
                        print(f"Cluster {cluster_id}: {percentage:.2f}% (Color: {kmeans.cluster_centers_[cluster_id].astype(int)})")
                    
                    # Create a label image for finding boundaries
                    label_img = np.zeros(frame.shape[:2], dtype=np.uint8)
                    label_img[binary_mask == 1] = labels + 1  # +1 to avoid 0 (background)
                    
                    # Find boundaries between clusters using morphological operations
                    kernel = np.ones((5, 5), np.uint8)
                    dilated = cv2.dilate(label_img, kernel, iterations=5)
                    eroded = cv2.erode(label_img, kernel, iterations=5)
                    boundary = cv2.absdiff(dilated, eroded)
                    boundary = (boundary > 0).astype(np.uint8) * 255
                    
                    # Start with original frame
                    result_img = frame.copy()
                    
                    # Draw thick boundary line between clusters on original frame
                    result_img[boundary == 255] = [0, 255, 255]  # Yellow boundary line
                    
                    # Add text with percentages
                    y_offset = 30
                    for cluster_name, percentage in percentages.items():
                        text = f"{cluster_name}: {percentage:.2f}%"
                        cv2.putText(result_img, text, (10, y_offset), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        y_offset += 30
                    
                    cv2.imshow("YOLO Inference", result_img)
                else:
                    cv2.imshow("YOLO Inference", masks)
            else:
                cv2.imshow("YOLO Inference", masks)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()

