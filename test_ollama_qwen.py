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

