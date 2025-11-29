import pyrealsense2 as rs
import cv2
import numpy as np

# ตั้งค่า Pipeline
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.infrared, 1, 848, 480, rs.format.y8, 30)

pipeline.start(config)

# ROI drawing variables
drawing = False
ix, iy = -1, -1
ex, ey = -1, -1
roi_rect = None

# Mouse callback function
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, ex, ey, drawing, roi_rect
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        ex, ey = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            ex, ey = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        ex, ey = x, y
        roi_rect = (min(ix, ex), min(iy, ey), abs(ex - ix), abs(ey - iy))

cv2.namedWindow('RealSense IR Stream')
cv2.setMouseCallback('RealSense IR Stream', draw_rectangle)

try:
    while True:
        frames = pipeline.wait_for_frames()
        ir_frame = frames.get_infrared_frame(1) # ดึงภาพจาก IR กล้องซ้าย
        
        if not ir_frame:
            continue

        ir_image = np.asanyarray(ir_frame.get_data())
        display_image = ir_image.copy()

        # Draw ROI rectangle if being drawn or set
        if drawing or roi_rect:
            x0 = ix if drawing else roi_rect[0]
            y0 = iy if drawing else roi_rect[1]
            x1 = ex if drawing else roi_rect[0] + roi_rect[2]
            y1 = ey if drawing else roi_rect[1] + roi_rect[3]
            cv2.rectangle(display_image, (x0, y0), (x1, y1), (255, 255, 255), 2)

        # Use ROI if set, otherwise use full image
        if roi_rect:
            x, y, w, h = roi_rect
            roi = ir_image[y:y+h, x:x+w]
            brightness = np.mean(roi)
        else:
            brightness = np.mean(ir_image[:,:])
        status = "Veg Oil" if brightness > 100 else "Motor Oil"

        cv2.putText(display_image, f"Status: {status} ({brightness:.1f})", (50,50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.imshow('RealSense IR Stream', display_image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()