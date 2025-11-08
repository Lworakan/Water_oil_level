import ollama
import base64
import cv2
import re

# Read the image
image_path = 'screenshots/screenshot_20251108_210827.jpg'
img = cv2.imread(image_path)

with open(image_path, 'rb') as f:
    image_bytes = f.read()
encoded_image = base64.b64encode(image_bytes).decode('utf-8')

# Define your prompt
prompt = "What is in this image?, please label two main objects [UCO liquid base,upper laye] as list of x1 y1 x2 y2 coordinates."

print(f"Sending prompt to qwen3-vl model...")
print(f"Prompt: {prompt}\n")

# Send the request to Ollama
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
    
    # Print full response for debugging
    print("Full Response Object:")
    print(response)
    print("\n" + "="*50 + "\n")
    
    # Print the response content
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

# Parse coordinates from response - handle multiple formats
import re

bboxes = []

# Try format: x1=435 y1=480 x2=585 y2=850
pattern1 = r'x1=(\d+)\s+y1=(\d+)\s+x2=(\d+)\s+y2=(\d+)'
matches1 = re.findall(pattern1, content)
for match in matches1:
    bbox = [float(x) for x in match]
    bboxes.append(bbox)

# Try format: [430, 510, 590, 830]
if not bboxes:
    pattern2 = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
    matches2 = re.findall(pattern2, content)
    for match in matches2:
        bbox = [float(x) for x in match]
        bboxes.append(bbox)

# Try simple number extraction as fallback
if not bboxes:
    numbers = [float(x) for x in content.split() if x.replace('.','').replace('-','').isdigit()]
    for i in range(0, len(numbers), 4):
        if i + 3 < len(numbers):
            bbox = numbers[i:i+4]
            bboxes.append(bbox)

for idx, bbox in enumerate(bboxes, 1):
    print(f"Bounding Box {idx}: {bbox}")

if not bboxes:
    print("\n⚠️  WARNING: No bounding boxes found in response!")
    print("The qwen3 model may not be responding correctly.")
    print("Please check if the model is running: ollama list")
    print("You may need to pull the model: ollama pull qwen2-vl:7b")
    exit(1)

# Draw bounding boxes on image
colors = [(0, 255, 0), (255, 0, 255), (0, 255, 255), (255, 165, 0)]  # Green, Magenta, Yellow, Orange
labels = ["Upper Layer", "UCO Base", "Object 3", "Object 4"]

result_img = img.copy()

for idx, bbox in enumerate(bboxes):
    x1, y1, x2, y2 = map(int, bbox)
    color = colors[idx % len(colors)]
    label = labels[idx % len(labels)]
    
    # Draw rectangle
    cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 3)
    
    # Draw label with background
    label_text = f"{label}: [{x1},{y1},{x2},{y2}]"
    (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(result_img, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
    cv2.putText(result_img, label_text, (x1, y1 - 5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# Save the result
output_path = 'screenshots/qwen3_bbox_result.jpg'
cv2.imwrite(output_path, result_img)
print(f"\nResult saved to: {output_path}")

# Display the result
cv2.imshow("Qwen3 Bounding Boxes", result_img)
print("Press any key to close the window...")
cv2.waitKey(0)
cv2.destroyAllWindows()