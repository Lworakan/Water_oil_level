import cv2
import numpy as np
from sklearn.cluster import KMeans

from ultralytics import YOLO
from ultralytics import SAM

model = YOLO("yolo11n.pt")
model_seg = SAM("sam2.1_b.pt")

image_path = 'screenshots/screenshot_20251108_210827.jpg'
print(f"Loading image from: {image_path}")
frame = cv2.imread(image_path)

if frame is not None:
    print(f"Image loaded successfully. Shape: {frame.shape}")
    
    if True:
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
            
            if hasattr(mask, 'masks') and mask.masks is not None:
                seg_mask = mask.masks.data[0].cpu().numpy()  # Get first mask
                
                if seg_mask.shape != frame.shape[:2]:
                    seg_mask = cv2.resize(seg_mask, (frame.shape[1], frame.shape[0]))
                binary_mask = (seg_mask > 0.5).astype(np.uint8)
                
                masked_pixels = frame[binary_mask == 1]
                
                if len(masked_pixels) > 0:
                    pixels_reshaped = masked_pixels.reshape(-1, 3)
                    
                    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(pixels_reshaped)
                    unique, counts = np.unique(labels, return_counts=True)
                    total_pixels = len(labels)
                    
                    percentages = {}
                    for cluster_id, count in zip(unique, counts):
                        percentage = (count / total_pixels) * 100
                        percentages[f'Cluster {cluster_id}'] = percentage
                        print(f"Cluster {cluster_id}: {percentage:.2f}% (Color: {kmeans.cluster_centers_[cluster_id].astype(int)})")
                    cluster_colors = [
                        [0, 255, 255],  
                        [255, 0, 255]   
                    ]
                    
                    result_img = frame.copy()
                    
                    overlay = result_img.copy()
                    
                    cluster_mask_img = np.zeros_like(frame)
                    pixel_idx = 0
                    for i in range(frame.shape[0]):
                        for j in range(frame.shape[1]):
                            if binary_mask[i, j] == 1:
                                cluster_id = labels[pixel_idx]
                                cluster_mask_img[i, j] = cluster_colors[cluster_id]
                                overlay[i, j] = cluster_colors[cluster_id]
                                pixel_idx += 1
                    
                    alpha = 0.5
                    result_img = cv2.addWeighted(result_img, 1 - alpha, overlay, alpha, 0)
                    label_img = np.zeros(frame.shape[:2], dtype=np.uint8)
                    label_img[binary_mask == 1] = labels + 1
                    kernel = np.ones((3, 3), np.uint8)
                    dilated = cv2.dilate(label_img, kernel, iterations=2)
                    eroded = cv2.erode(label_img, kernel, iterations=2)
                    boundary = cv2.absdiff(dilated, eroded)
                    boundary = (boundary > 0).astype(np.uint8) * 255
                    result_img[boundary == 255] = [255, 255, 255]  # White boundary line
                    
                    y_offset = 30
                    for cluster_name, percentage in percentages.items():
                        text = f"{cluster_name}: {percentage:.2f}%"
                        cv2.putText(result_img, text, (10, y_offset), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        y_offset += 30
                    
                    output_path = 'screenshots/bottle_seg_cluster_result.jpg'
                    cv2.imwrite(output_path, result_img)
                    print(f"\nResult saved to: {output_path}")
                    
                    cv2.imshow("YOLO Inference - Press any key to close", result_img)
                    cv2.waitKey(0)
                else:
                    cv2.imshow("YOLO Inference - Press any key to close", masks)
                    cv2.waitKey(0)
            else:
                cv2.imshow("YOLO Inference - Press any key to close", masks)
                cv2.waitKey(0)
else:
    print(f"Error: Could not load image from {image_path}")

cv2.destroyAllWindows()

