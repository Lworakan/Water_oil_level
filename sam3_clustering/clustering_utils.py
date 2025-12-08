import cv2
import numpy as np
import torch
from sklearn.cluster import KMeans

def apply_clustering(image_cv, masks, n_clusters=2):
    """
    Apply K-Means clustering to the segmented area.
    """
    if len(masks) == 0:
        print("No masks found.")
        return image_cv

    combined_mask = torch.any(masks, dim=0).cpu().numpy().astype(np.uint8)
    
    if combined_mask.shape != image_cv.shape[:2]:
        combined_mask = cv2.resize(combined_mask, (image_cv.shape[1], image_cv.shape[0]), interpolation=cv2.INTER_NEAREST)

    masked_pixels = image_cv[combined_mask == 1]
    
    if len(masked_pixels) == 0:
        print("No pixels in mask.")
        return image_cv

    print(f"Clustering {len(masked_pixels)} pixels into {n_clusters} clusters...")
    pixels_reshaped = masked_pixels.reshape(-1, 3)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels_reshaped)
    
    unique, counts = np.unique(labels, return_counts=True)
    total_pixels = len(labels)
    percentages = {}
    
    print("Cluster statistics:")
    for cluster_id, count in zip(unique, counts):
        percentage = (count / total_pixels) * 100
        percentages[cluster_id] = percentage
        center_color = kmeans.cluster_centers_[cluster_id].astype(int)
        print(f"  Cluster {cluster_id}: {percentage:.2f}% (Center Color BGR: {center_color})")

    result_img = image_cv.copy()
    overlay = result_img.copy()
    
  
    vis_colors = [
        [0, 255, 255],  # Yellow
        [255, 0, 255],  # Magenta
        [255, 255, 0],  # Cyan
        [0, 255, 0],    # Green
    ]
    
   
    full_labels = np.full(image_cv.shape[:2], -1, dtype=int)
    full_labels[combined_mask == 1] = labels
    
    for cluster_id in unique:
        color = vis_colors[cluster_id % len(vis_colors)]
        overlay[full_labels == cluster_id] = color
        
    alpha = 0.5
    result_img = cv2.addWeighted(result_img, 1 - alpha, overlay, alpha, 0)
    
   
    label_map = np.zeros(image_cv.shape[:2], dtype=np.uint8)
    label_map[combined_mask == 1] = labels + 1
    
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(label_map, kernel, iterations=2)
    eroded = cv2.erode(label_map, kernel, iterations=2)
    boundary = cv2.absdiff(dilated, eroded)
    
    result_img[boundary > 0] = [255, 255, 255]

    y_offset = 30
    for cluster_id, percentage in percentages.items():
        text = f"Cluster {cluster_id}: {percentage:.2f}%"
        cv2.putText(result_img, text, (10, y_offset), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30
        
    return result_img
