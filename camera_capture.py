import cv2
import os
from datetime import datetime

def main():
    # Open the camera (0 is usually the default camera)
    cap = cv2.VideoCapture(0)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Camera opened successfully!")
    print("Press 's' to take a screenshot")
    print("Press 'q' to quit")
    
    # Create a folder to save screenshots if it doesn't exist
    save_folder = "screenshots"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Can't receive frame")
            break
        
        # Display the frame
        cv2.imshow('Camera - Press "s" to capture, "q" to quit', frame)
        
        # Wait for key press (1ms delay)
        key = cv2.waitKey(1) & 0xFF
        
        # If 's' is pressed, save the screenshot
        if key == ord('s'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(save_folder, f"screenshot_{timestamp}.jpg")
            cv2.imwrite(filename, frame)
            print(f"Screenshot saved: {filename}")
        
        # If 'q' is pressed, quit
        elif key == ord('q'):
            print("Quitting...")
            break
    
    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
