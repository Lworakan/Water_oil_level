import cv2
import os
import time
from datetime import datetime

def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def view_camera_mode(camera_index=0):
    print(f"Starting Camera View Mode (Camera Index: {camera_index})...")
    print("Press 'q' to quit.")
    
    # Use CAP_DSHOW on Windows to avoid delays/hanging
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    
    # Set resolution to 1920x1080 to ensure full frame
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        cv2.imshow('Camera View Mode', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def record_video_mode(camera_index=0, output_dir='datasets/videos'):
    create_dir(output_dir)
    print(f"Starting Video Recording Mode (Camera Index: {camera_index})...")
    print(f"Saving videos to: {output_dir}")
    print("Press 'r' to toggle recording.")
    print("Press 'q' to quit.")

    # Use CAP_DSHOW on Windows to avoid delays/hanging
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

    # Set resolution to 1920x1080 to ensure full frame
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = None
    is_recording = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        if is_recording:
            if out is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(output_dir, f"video_{timestamp}.avi")
                # Get frame width and height
                frame_width = int(cap.get(3))
                frame_height = int(cap.get(4))
                out = cv2.VideoWriter(filename, fourcc, 20.0, (frame_width, frame_height))
                print(f"Started recording: {filename}")
            
            out.write(frame)
            # Add a visual indicator for recording
            cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1) 
            cv2.putText(frame, "REC", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow('Video Recording Mode', frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('r'):
            is_recording = not is_recording
            if not is_recording and out is not None:
                out.release()
                out = None
                print("Stopped recording.")

    if out is not None:
        out.release()
    cap.release()
    cv2.destroyAllWindows()

def snapshot_image_mode(camera_index=0, output_dir='datasets/images'):
    create_dir(output_dir)
    print(f"Starting Snapshot Image Mode (Camera Index: {camera_index})...")
    print(f"Saving images to: {output_dir}")
    print("Press 's' or 'SPACE' to take a snapshot.")
    print("Press 'q' to quit.")

    # Use CAP_DSHOW on Windows to avoid delays/hanging
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

    # Set resolution to 1920x1080 to ensure full frame
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        cv2.imshow('Snapshot Image Mode', frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('s') or key == 32: # 32 is SPACE
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(output_dir, f"img_{timestamp}.jpg")
            cv2.imwrite(filename, frame)
            print(f"Snapshot saved: {filename}")
            # Flash effect
            white_frame = frame.copy()
            white_frame[:] = 255
            cv2.imshow('Snapshot Image Mode', white_frame)
            cv2.waitKey(50)

    cap.release()
    cv2.destroyAllWindows()

def main():
    # Allow user to specify camera index
    cam_idx_str = input("Enter camera index (default 0): ")
    camera_index = int(cam_idx_str) if cam_idx_str.isdigit() else 0

    while True:
        print("\n--- Data Collection Tool ---")
        print("1. Open Camera (Real-time View)")
        print("2. Record Video Mode")
        print("3. Snapshot Image Mode")
        print("4. Exit")
        
        choice = input("Select a mode (1-4): ")

        if choice == '1':
            view_camera_mode(camera_index)
        elif choice == '2':
            record_video_mode(camera_index)
        elif choice == '3':
            snapshot_image_mode(camera_index)
        elif choice == '4':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
