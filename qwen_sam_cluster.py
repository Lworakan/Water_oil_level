import ollama
import base64
import cv2
import re

from ultralytics import YOLO
from ultralytics import SAM
# Read the image
image_path = 'screenshots/screenshot_20251108_210827.jpg'
img = cv2.imread(image_path)
model_seg = SAM("sam2.1_b.pt")

with open(image_path, 'rb') as f:
    image_bytes = f.read()
encoded_image = base64.b64encode(image_bytes).decode('utf-8')

# Define your prompt
# prompt = "Locate every instance that belongs to the following categories objects [UCO liquid base,upper layer emtry space in the bottle] as list of x1 y1 x2 y2 coordinates."
prompt = "Locate every instance that belongs to the following category: 'bottle'. Report bbox coordinates in JSON format."


print(f"Sending prompt to qwen3-vl model...")
print(f"Prompt: {prompt}\n")

try:
    response = ollama.chat(
        model='qwen3-vl',
        messages=[
            {
                'role': 'user',
                'content': prompt,
                'images': [encoded_image]
            }
        ]
    )
    
    print("Full Response Object:")
    print(response)
    print("\n" + "="*50 + "\n")
    
    content = response['message']['content']
    print("Qwen3 Response Content:")
    print(repr(content))  # Use repr to show hidden characters
    print(content)
    print("\n" + "="*50 + "\n")
    
except Exception as e:
    print(f"Error calling Ollama: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

import re
import json

bboxes = []

# Try to parse as JSON first
try:
    # Extract JSON from markdown code blocks if present
    json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # Try to find JSON object directly
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            json_str = content
    
    data = json.loads(json_str)
    
    # Handle different JSON structures
    if isinstance(data, dict):
        # Try common keys for bounding boxes
        for key in ['bbox', 'bboxes', 'boxes', 'coordinates', 'bottle', 'bottles', 'objects']:
            if key in data:
                bbox_data = data[key]
                if isinstance(bbox_data, list):
                    # Could be list of boxes or single box
                    if len(bbox_data) == 4 and all(isinstance(x, (int, float)) for x in bbox_data):
                        # Single box: [x1, y1, x2, y2]
                        bboxes.append([float(x) for x in bbox_data])
                    else:
                        # Multiple boxes
                        for item in bbox_data:
                            if isinstance(item, list) and len(item) == 4:
                                bboxes.append([float(x) for x in item])
                            elif isinstance(item, dict):
                                # Extract coordinates from dict
                                if all(k in item for k in ['x1', 'y1', 'x2', 'y2']):
                                    bboxes.append([float(item['x1']), float(item['y1']), 
                                                 float(item['x2']), float(item['y2'])])
                                elif 'bbox' in item:
                                    bboxes.append([float(x) for x in item['bbox']])
                break
    elif isinstance(data, list):
        # List of boxes
        for item in data:
            if isinstance(item, list) and len(item) == 4:
                bboxes.append([float(x) for x in item])
            elif isinstance(item, dict) and 'bbox' in item:
                bboxes.append([float(x) for x in item['bbox']])
    
    print(f"Parsed {len(bboxes)} bounding boxes from JSON")
    
except (json.JSONDecodeError, ValueError) as e:
    print(f"JSON parsing failed: {e}, trying regex patterns...")

# Fallback to regex patterns if JSON parsing failed
if not bboxes:
    pattern1 = r'x1=(\d+)\s+y1=(\d+)\s+x2=(\d+)\s+y2=(\d+)'
    matches1 = re.findall(pattern1, content)
    for match in matches1:
        bbox = [float(x) for x in match]
        bboxes.append(bbox)

if not bboxes:
    pattern2 = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
    matches2 = re.findall(pattern2, content)
    for match in matches2:
        bbox = [float(x) for x in match]
        bboxes.append(bbox)

if not bboxes:
    numbers = [float(x) for x in content.split() if x.replace('.','').replace('-','').isdigit()]
    for i in range(0, len(numbers), 4):
        if i + 3 < len(numbers):
            bbox = numbers[i:i+4]
            bboxes.append(bbox)

if not bboxes:
    print("\n⚠️  WARNING: No bounding boxes found in response!")
    print("Please check the qwen3 model response format.")
    exit(1)

print(f"\nOriginal image dimensions: {img.shape[1]} x {img.shape[0]} (W x H)")
print(f"Found {len(bboxes)} bounding box(es) from qwen3")

# Scale bounding boxes from qwen3's normalized coordinates to actual image size
# Qwen3-VL typically uses 0-1000 range, so we need to scale to actual image dimensions
scaled_bboxes = []
for idx, bbox in enumerate(bboxes, 1):
    x1, y1, x2, y2 = bbox
    
    # Check if coordinates are in normalized range (0-1000)
    if max(bbox) <= 1000:
        # Scale from 0-1000 range to actual image dimensions
        img_width = img.shape[1]
        img_height = img.shape[0]
        
        x1_scaled = (x1 / 1000.0) * img_width
        y1_scaled = (y1 / 1000.0) * img_height
        x2_scaled = (x2 / 1000.0) * img_width
        y2_scaled = (y2 / 1000.0) * img_height
        
        scaled_bbox = [x1_scaled, y1_scaled, x2_scaled, y2_scaled]
        scaled_bboxes.append(scaled_bbox)
        
        print(f"Bounding Box {idx} (original): {bbox}")
        print(f"Bounding Box {idx} (scaled):   {[int(x) for x in scaled_bbox]}")
    else:
        scaled_bboxes.append(bbox)
        print(f"Bounding Box {idx} (already scaled): {bbox}")

result_img = img.copy()
colors = [(0, 255, 0), (255, 0, 255), (0, 255, 255), (255, 165, 0)]
labels = ["Bottle", "Object 2", "Object 3", "Object 4"]

for idx, bbox in enumerate(scaled_bboxes):
    x1, y1, x2, y2 = map(int, bbox)
    color = colors[idx % len(colors)]
    label = labels[idx % len(labels)]
    
    cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 3)
    cv2.putText(result_img, f"{label}: [{x1},{y1},{x2},{y2}]", (x1, y1 - 10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Save and show bbox visualization
bbox_output = 'screenshots/qwen3_bbox_scaled.jpg'
cv2.imwrite(bbox_output, result_img)
print(f"\nBounding box visualization saved to: {bbox_output}")

cv2.imshow("Qwen3 Bounding Boxes (Scaled)", result_img)
print("Press any key to continue to segmentation...")
cv2.waitKey(0)
cv2.destroyAllWindows()

# Use scaled bboxes for segmentation
print("\n" + "="*50)
print("Running SAM2 Segmentation on detected regions...")
print("="*50 + "\n")

esults_seg = model_seg(img, bboxes=scaled_bboxes, stream=True)

for mask_result in esults_seg:
    # Show basic SAM2 segmentation
    masks_viz = mask_result.plot()
    cv2.imshow("SAM2 Segmentation", masks_viz)
    cv2.imwrite('screenshots/qwen_sam_segmentation.jpg', masks_viz)
    print("SAM2 segmentation saved to: screenshots/qwen_sam_segmentation.jpg")
    
    # Now apply clustering to each segmented region
    if hasattr(mask_result, 'masks') and mask_result.masks is not None:
        print(f"\nFound {len(mask_result.masks.data)} mask(s)")
        
        # Import clustering libraries
        import numpy as np
        from sklearn.cluster import KMeans
        
        result_img = img.copy()
        
        # Define colors for different masks
        mask_colors = [
            [0, 255, 255],    # Yellow for first mask
            [255, 0, 255]     # Magenta for second mask
        ]
        
        # First pass: calculate total pixels across all masks
        total_pixels_all_masks = 0
        mask_pixel_counts = []
        
        for seg_mask in mask_result.masks.data:
            seg_mask_np = seg_mask.cpu().numpy()
            if seg_mask_np.shape != img.shape[:2]:
                seg_mask_np = cv2.resize(seg_mask_np, (img.shape[1], img.shape[0]))
            binary_mask = (seg_mask_np > 0.5).astype(np.uint8)
            pixel_count = np.sum(binary_mask)
            mask_pixel_counts.append(pixel_count)
            total_pixels_all_masks += pixel_count
        
        print(f"Total pixels across all masks: {total_pixels_all_masks}")
        
        # Process each mask separately
        for mask_idx, seg_mask in enumerate(mask_result.masks.data):
            print(f"\n--- Processing Mask {mask_idx + 1} ---")
            
            seg_mask_np = seg_mask.cpu().numpy()
            
            # Resize mask to match frame size if needed
            if seg_mask_np.shape != img.shape[:2]:
                seg_mask_np = cv2.resize(seg_mask_np, (img.shape[1], img.shape[0]))
            
            # Convert to binary mask
            binary_mask = (seg_mask_np > 0.5).astype(np.uint8)
            
            # Extract pixels within the mask
            masked_pixels = img[binary_mask == 1]
            
            if len(masked_pixels) > 10:  # Need enough pixels for clustering
                # Color this mask region with semi-transparent overlay
                overlay = result_img.copy()
                color = mask_colors[mask_idx % len(mask_colors)]
                overlay[binary_mask == 1] = color
                result_img = cv2.addWeighted(result_img, 0.7, overlay, 0.3, 0)
                
                # Calculate percentage of this mask relative to total
                percentage = (mask_pixel_counts[mask_idx] / total_pixels_all_masks) * 100
                
                # Calculate average color
                avg_color = np.mean(masked_pixels, axis=0)
                print(f"Mask {mask_idx + 1} - Average Color (BGR): {avg_color.astype(int)}")
                print(f"Mask {mask_idx + 1} - Number of pixels: {len(masked_pixels)}")
                print(f"Mask {mask_idx + 1} - Percentage: {percentage:.2f}%")
                
                # Add label on image with percentage
                y_positions = [30, 60, 90]
                region_names = ["Upper Layer", "UCO Base", "Region 3"]
                region_name = region_names[mask_idx % len(region_names)]
                label_text = f"{region_name}: {percentage:.2f}% ({len(masked_pixels)} px)"
                cv2.putText(result_img, label_text, (10, y_positions[mask_idx % len(y_positions)]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Save and display result
        output_path = 'screenshots/qwen_sam_cluster_result.jpg'
        cv2.imwrite(output_path, result_img)
        print(f"\nClustered segmentation result saved to: {output_path}")
        
        cv2.imshow("Qwen3 + SAM2 - Two Region Segmentation", result_img)
        print("Press any key to close...")
        cv2.waitKey(0)
    else:
        cv2.imshow("SAM2 Result", masks_viz)
        cv2.waitKey(0)

cv2.destroyAllWindows()
print("\n✅ Processing complete!")