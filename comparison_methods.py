import cv2
import numpy as np
import ollama
import base64
from ultralytics import YOLO
from ultralytics import SAM
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Load models
model_yolo = YOLO("yolo11n.pt")
model_sam = SAM("sam2.1_b.pt")

def get_qwen3_bbox(frame):
    """Get bounding box(es) from qwen3-vl model"""
    # Encode frame to base64
    _, buffer = cv2.imencode('.jpg', frame)
    encoded_image = base64.b64encode(buffer).decode('utf-8')
    
    # Define prompt for bounding box detection
    prompt = "What is in this image?, please label two main objects [UCO liquid base,upper laye] as list of x1 y1 x2 y2 coordinates."

    
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
        
        # Parse response to extract coordinates
        content = response['message']['content']
        print(f"Qwen3 response: {content}")
        
        # Simple parsing - extract all numbers
        numbers = [float(x) for x in content.split() if x.replace('.','').replace('-','').isdigit()]
        
        # Return all bounding boxes (each bbox has 4 coordinates)
        bboxes = []
        for i in range(0, len(numbers), 4):
            if i + 3 < len(numbers):
                bboxes.append(numbers[i:i+4])
        
        if len(bboxes) > 0:
            return bboxes
            
    except Exception as e:
        print(f"Qwen3 error: {e}")
    
    return None

def method_a_yolo_detection(frame):
    """A: YOLO Detection (Image-only)"""
    results = model_yolo(frame, classes=[61], conf=0.25)
    annotated = results[0].plot()
    return annotated

def method_b_yolo_sam(frame):
    """B: YOLO Detection then SAM2 Segmentation"""
    results = model_yolo(frame, classes=[61], conf=0.25)
    
    for result in results:
        xyxy = result.boxes.xyxy   
        if len(xyxy) > 0:
            x1, y1, x2, y2 = map(float, xyxy[0].tolist())
            bounding_box = [x1, y1, x2, y2]
            
            annotated_frame = results[0].plot()
            results_seg = model_sam(annotated_frame, bboxes=bounding_box, stream=True)
            
            for mask in results_seg:
                return mask.plot()
    
    return results[0].plot()

def method_c_sam_only(frame):
    """C: Segmentation-only by SAM2"""
    # SAM2 without bounding box - will segment all detected objects
    results_seg = model_sam(frame, stream=True)
    
    for mask in results_seg:
        return mask.plot()
    
    return frame

def method_d_qwen3_bbox(frame):
    """D: qwen3 Bounding Box (xy from qwen3)"""
    bboxes = get_qwen3_bbox(frame)
    result_img = frame.copy()
    
    if bboxes:
        colors = [(0, 255, 0), (255, 0, 255)]  # Green for first, Magenta for second
        labels = ["Upper Layer", "UCO Base"]
        
        for idx, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = map(int, bbox)
            color = colors[idx % len(colors)]
            label = labels[idx % len(labels)]
            
            cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 3)
            cv2.putText(result_img, f"Qwen3: {label}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return result_img

def method_e_qwen3_sam(frame):
    """E: qwen3-Guided Segmentation (inside qwen3 bbox)"""
    bboxes = get_qwen3_bbox(frame)
    
    if bboxes:
        # Use all bounding boxes for segmentation
        bounding_boxes = [[float(x) for x in bbox] for bbox in bboxes]
        
        results_seg = model_sam(frame, bboxes=bounding_boxes, stream=True)
        
        for mask in results_seg:
            return mask.plot()
    
    return frame

def create_comparison_figure(frame):
    """Create comparison figure with all methods"""
    
    print("Running Method A: YOLO Detection...")
    img_a = method_a_yolo_detection(frame)
    
    print("Running Method B: YOLO + SAM2...")
    img_b = method_b_yolo_sam(frame)
    
    print("Running Method C: SAM2 Only...")
    img_c = method_c_sam_only(frame)
    
    print("Running Method D: Qwen3 Bounding Box...")
    img_d = method_d_qwen3_bbox(frame)
    
    print("Running Method E: Qwen3 + SAM2...")
    img_e = method_e_qwen3_sam(frame)
    
    # Create figure with 2x3 grid
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Convert BGR to RGB for matplotlib
    images = [
        (cv2.cvtColor(img_a, cv2.COLOR_BGR2RGB), "A: YOLO Detection\n(Image-only)"),
        (cv2.cvtColor(img_b, cv2.COLOR_BGR2RGB), "B: YOLO Detection\nthen SAM2 Segmentation"),
        (cv2.cvtColor(img_c, cv2.COLOR_BGR2RGB), "C: Segmentation-only\nby SAM2"),
        (cv2.cvtColor(img_d, cv2.COLOR_BGR2RGB), "D: qwen3 Bounding Box\n(xy from qwen3)"),
        (cv2.cvtColor(img_e, cv2.COLOR_BGR2RGB), "E: qwen3-Guided Segmentation\n(inside qwen3 bbox)")
    ]
    
    # Plot images in 2x3 grid
    positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
    
    for idx, (img, title) in enumerate(images):
        row, col = positions[idx]
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(img)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')
    
    # Add main title
    fig.suptitle('Comparison of Detection, Segmentation, and Qwen3-Guided Methods',
                fontsize=16, fontweight='bold', y=0.98)
    
    # Save figure
    plt.savefig('screenshots/comparison_methods.png', dpi=150, bbox_inches='tight')
    print("Comparison figure saved to: screenshots/comparison_methods.png")
    
    plt.show()

def main():
    """Main function to run comparison"""
    # Open camera or load image
    print("Choose input source:")
    print("1. Camera (real-time)")
    print("2. Image file")
    choice = input("Enter choice (1 or 2): ")
    
    if choice == "1":
        # Use camera
        cap = cv2.VideoCapture(0)
        print("Press SPACE to capture frame for comparison, or 'q' to quit")
        
        while True:
            success, frame = cap.read()
            if success:
                cv2.imshow("Camera - Press SPACE to capture", frame)
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord(' '):  # Space bar
                    print("\nCapturing frame for comparison...")
                    create_comparison_figure(frame)
                    break
                elif key == ord('q'):
                    break
        
        cap.release()
        cv2.destroyAllWindows()
        
    else:
        # Use image file
        image_path = input("Enter image path (or press Enter for default): ")
        if not image_path:
            image_path = "screenshots/screenshot_20251108_210827.jpg"
        
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error: Could not load image from {image_path}")
            return
        
        print(f"Loaded image from: {image_path}")
        create_comparison_figure(frame)

if __name__ == "__main__":
    main()
