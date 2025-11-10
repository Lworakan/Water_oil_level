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
# prompt = "What is in this image?, please label two main objects [UCO liquid base,upper laye] as list of x1 y1 x2 y2 coordinates."
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
        # Already in pixel coordinates
        scaled_bboxes.append(bbox)
        print(f"Bounding Box {idx} (already scaled): {bbox}")

# Draw scaled bounding boxes on image for verification
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
esults_seg = model_seg(image_path, bboxes=scaled_bboxes, stream=True)
for mask in esults_seg:
    masks = mask.plot()
cv2.imshow("YOLO Inference", masks)
cv2.waitKey(0)
cv2.destroyAllWindows()