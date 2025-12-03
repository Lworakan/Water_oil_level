import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import os
import csv
from datetime import datetime
import time
from PIL import Image, ImageTk
import pandas as pd
import numpy as np
import pyrealsense2 as rs

# --- CONFIGURATION ---
DATASET_ROOT = "Oil_Quality_Dataset_2025"
RAW_DIR = os.path.join(DATASET_ROOT, "01_Raw_Recordings")
MASTER_CSV_DIR = os.path.join(DATASET_ROOT, "02_Metadata_CSV")
FRAME_LOG_DIR = os.path.join(DATASET_ROOT, "03_Frame_Data_Logs") # <--- NEW FOLDER

MASTER_CSV_FILE = os.path.join(MASTER_CSV_DIR, "master_log.csv")

WEBCAM_INDEX = 0
ROTATE_PREVIEW = True 
ROTATE_DIR = cv2.ROTATE_90_COUNTERCLOCKWISE

# Create all folders
for folder in [RAW_DIR, MASTER_CSV_DIR, FRAME_LOG_DIR]:
    os.makedirs(folder, exist_ok=True)

# Create Master CSV Header
if not os.path.exists(MASTER_CSV_FILE):
    headers = [
        "filename_front_bag", "filename_top_mp4", "filename_frame_log_csv", # Added pointer to detailed log
        "timestamp", "phase", "bottle_id", "content_type", "volume", "note", "plastic_type", 
        "duration_sec", 
        "final_bbox_coords", "final_polygon_points", 
        "avg_ir_bbox_overall", "avg_ir_poly_overall"
    ]
    pd.DataFrame(columns=headers).to_csv(MASTER_CSV_FILE, index=False)

class TimeSeriesCollectorApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        
        # --- 1. SENSORS ---
        self.rs_pipeline = rs.pipeline()
        self.rs_config = rs.config()
        self.rs_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.rs_config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
        
        try:
            profile = self.rs_pipeline.start(self.rs_config)
            self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        except Exception as e:
            messagebox.showerror("Error", f"RealSense Error: {e}")

        self.cap_webcam = cv2.VideoCapture(WEBCAM_INDEX, cv2.CAP_DSHOW)
        self.cap_webcam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap_webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Recording State
        self.is_recording = False
        self.out_webcam = None
        self.start_time = None
        self.current_filename_base = ""
        
        # Frame Logging Variables
        self.frame_log_file = None
        self.frame_log_writer = None
        self.frame_count = 0
        self.current_frame_log_name = ""

        # Annotation Variables
        self.bbox_start = None; self.bbox_end = None
        self.val_bbox_ir = 0; self.val_bbox_depth = 0
        self.saved_bbox_coords = (0,0,0,0)

        self.poly_points_canvas = []
        self.val_poly_ir = 0
        self.saved_poly_coords = []

        # --- 2. GUI LAYOUT ---
        self.control_frame = tk.Frame(window, padx=10, pady=10)
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y)
        self.create_controls()

        self.video_frame = tk.Frame(window, bg="#333333")
        self.video_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        rs_w, rs_h = (360, 480) if ROTATE_PREVIEW else (640, 480)
        
        tk.Label(self.video_frame, text="FRONT: Left Click=Box | Right Click=Poly", bg="black", fg="white").grid(row=0, column=0)
        self.canvas_rs = tk.Canvas(self.video_frame, width=rs_w, height=rs_h, bg="black", cursor="cross")
        self.canvas_rs.grid(row=1, column=0, padx=10, pady=10)
        
        self.canvas_rs.bind("<ButtonPress-1>", self.on_bbox_start)
        self.canvas_rs.bind("<B1-Motion>", self.on_bbox_drag)
        self.canvas_rs.bind("<ButtonRelease-1>", self.on_bbox_end)
        self.canvas_rs.bind("<ButtonPress-3>", self.on_poly_add)
        self.canvas_rs.bind("<Double-Button-3>", self.on_poly_reset)

        tk.Label(self.video_frame, text="TOP VIEW", bg="black", fg="white").grid(row=0, column=1)
        self.canvas_web = tk.Canvas(self.video_frame, width=400, height=300, bg="black")
        self.canvas_web.grid(row=1, column=1, padx=10, pady=10)

        self.update_frames()
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_controls(self):
        self.vars = {}
        form_fields = [
            ("Phase", "Phase1_Pure", ("Phase1_Pure", "Phase2_Mixture")),
            ("Bottle ID", "01", None),
            ("Content", "Veg_Oil", ("Veg_Oil_Light", "Veg_Oil_Med", "Veg_Oil_Dark", "Motor_Oil", "Lard", "Water", "Mix")),
            ("Volume", "Full_100pct", ("Full_100pct", "Half_50pct", "Low_25pct", "Empty")), 
            ("Plastic", "PP_Clear", ("PP_Clear", "PP_Cloudy", "HDPE")),
            ("Note", "", None)
        ]

        for label, default, vals in form_fields:
            tk.Label(self.control_frame, text=f"{label}:", font=("Arial", 10, "bold")).pack(anchor="w")
            self.vars[label] = tk.StringVar(value=default)
            if vals:
                ttk.Combobox(self.control_frame, textvariable=self.vars[label], values=vals).pack(fill=tk.X)
            else:
                tk.Entry(self.control_frame, textvariable=self.vars[label]).pack(fill=tk.X)

        tk.Label(self.control_frame, text="--- LIVE IR ---", font=("Arial", 10, "bold"), fg="blue").pack(pady=(20,5))
        self.lbl_bbox_ir = tk.Label(self.control_frame, text="Box IR: -", font=("Consolas", 12), bg="black", fg="#00ff00")
        self.lbl_bbox_ir.pack(fill=tk.X, pady=1)
        self.lbl_poly_ir = tk.Label(self.control_frame, text="Poly IR: -", font=("Consolas", 12), bg="black", fg="#ff00ff")
        self.lbl_poly_ir.pack(fill=tk.X, pady=1)

        self.btn_record = tk.Button(self.control_frame, text="🔴 REC", bg="red", fg="white", 
                                    font=("Arial", 14, "bold"), command=self.toggle_recording)
        self.btn_record.pack(fill=tk.X, pady=20)
        self.lbl_status = tk.Label(self.control_frame, text="Ready", fg="green")
        self.lbl_status.pack()

    # --- MOUSE LOGIC (Same as before) ---
    def on_bbox_start(self, event): self.bbox_start = (event.x, event.y); self.bbox_end = (event.x, event.y)
    def on_bbox_drag(self, event): self.bbox_end = (event.x, event.y)
    def on_bbox_end(self, event): self.bbox_end = (event.x, event.y)
    def on_poly_add(self, event): self.poly_points_canvas.append((event.x, event.y))
    def on_poly_reset(self, event): self.poly_points_canvas = []; self.val_poly_ir = 0

    # --- MAIN LOOP ---
    def update_frames(self):
        try:
            frames = self.rs_pipeline.wait_for_frames(timeout_ms=100)
            if frames:
                color_frame = frames.get_color_frame()
                ir_frame = frames.get_infrared_frame()
                depth_frame = frames.get_depth_frame()
                
                img_color = np.asanyarray(color_frame.get_data())
                img_ir = np.asanyarray(ir_frame.get_data())
                img_depth = np.asanyarray(depth_frame.get_data())

                if ROTATE_PREVIEW:
                    img_color = cv2.rotate(img_color, ROTATE_DIR)
                    img_ir = cv2.rotate(img_ir, ROTATE_DIR)
                    img_depth = cv2.rotate(img_depth, ROTATE_DIR)

                disp_h, disp_w = img_color.shape[:2]
                target_w, target_h = (360, 480) if ROTATE_PREVIEW else (640, 480)
                scale_x = disp_w / target_w
                scale_y = disp_h / target_h
                img_disp = cv2.resize(img_color, (target_w, target_h))

                # --- 1. Calculate BBOX Stats ---
                if self.bbox_start and self.bbox_end:
                    x1, y1 = self.bbox_start; x2, y2 = self.bbox_end
                    rx1, rx2 = sorted([int(x1 * scale_x), int(x2 * scale_x)])
                    ry1, ry2 = sorted([int(y1 * scale_y), int(y2 * scale_y)])
                    rx1 = max(0, rx1); ry1 = max(0, ry1); rx2 = min(disp_w, rx2); ry2 = min(disp_h, ry2)

                    if rx2 > rx1 and ry2 > ry1:
                        roi_ir = img_ir[ry1:ry2, rx1:rx2]
                        roi_depth = img_depth[ry1:ry2, rx1:rx2]
                        self.val_bbox_ir = np.mean(roi_ir)
                        self.val_bbox_depth = np.mean(roi_depth) * self.depth_scale
                        self.saved_bbox_coords = (rx1, ry1, rx2, ry2)
                        cv2.rectangle(img_disp, self.bbox_start, self.bbox_end, (0, 255, 0), 2)
                    else:
                        self.val_bbox_ir = 0 # reset if box invalid

                # --- 2. Calculate POLY Stats ---
                if len(self.poly_points_canvas) > 2:
                    real_pts = [[int(pt[0] * scale_x), int(pt[1] * scale_y)] for pt in self.poly_points_canvas]
                    self.saved_poly_coords = real_pts
                    pts_np = np.array(real_pts, np.int32).reshape((-1, 1, 2))
                    mask = np.zeros(img_ir.shape, dtype=np.uint8)
                    cv2.fillPoly(mask, [pts_np], 255)
                    self.val_poly_ir = cv2.mean(img_ir, mask=mask)[0]
                    
                    pts_canvas = np.array(self.poly_points_canvas, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(img_disp, [pts_canvas], True, (255, 0, 255), 2)
                else:
                    self.val_poly_ir = 0

                # --- 3. RECORDING: FRAME-BY-FRAME LOG ---
                if self.is_recording and self.frame_log_writer:
                    elapsed = time.time() - self.start_time_epoch
                    row = [
                        self.frame_count,
                        f"{elapsed:.3f}",           # Time relative to start
                        f"{self.val_bbox_ir:.2f}",  # IR Value Box
                        f"{self.val_poly_ir:.2f}",  # IR Value Poly
                        f"{self.val_bbox_depth:.3f}", # Depth
                        str(self.saved_bbox_coords)   # Box coords (in case they move)
                    ]
                    self.frame_log_writer.writerow(row)
                    self.frame_count += 1

                # Update UI
                self.lbl_bbox_ir.config(text=f"Box IR: {self.val_bbox_ir:.1f}")
                self.lbl_poly_ir.config(text=f"Poly IR: {self.val_poly_ir:.1f}")
                img_rgb = cv2.cvtColor(img_disp, cv2.COLOR_BGR2RGB)
                self.photo_rs = ImageTk.PhotoImage(image=Image.fromarray(img_rgb))
                self.canvas_rs.create_image(0, 0, image=self.photo_rs, anchor=tk.NW)

        except Exception: pass

        if self.cap_webcam.isOpened():
            ret, web_frame = self.cap_webcam.read()
            if ret:
                if self.is_recording and self.out_webcam: self.out_webcam.write(web_frame)
                img_web_sm = cv2.resize(web_frame, (400, 300))
                self.photo_web = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(img_web_sm, cv2.COLOR_BGR2RGB)))
                self.canvas_web.create_image(0, 0, image=self.photo_web, anchor=tk.NW)

        self.window.after(15, self.update_frames)

    def toggle_recording(self):
        if not self.is_recording:
            # === START ===
            phase = self.vars["Phase"].get()
            bottle = self.vars["Bottle ID"].get()
            content = self.vars["Content"].get()
            vol = self.vars["Volume"].get()
            note = self.vars["Note"].get().replace(" ", "_")
            ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            self.current_filename_base = f"{phase}_B{bottle}_{content}_{vol}_{note}_{ts_str}"
            
            # Paths
            save_path = os.path.join(RAW_DIR, phase)
            os.makedirs(save_path, exist_ok=True)
            
            file_rs = os.path.join(save_path, self.current_filename_base + "_Front.bag")
            file_web = os.path.join(save_path, self.current_filename_base + "_Top.mp4")
            
            # --- START FRAME LOGGING CSV ---
            # Create a separate CSV for this specific recording session
            self.current_frame_log_name = self.current_filename_base + "_frames.csv"
            frame_log_path = os.path.join(FRAME_LOG_DIR, self.current_frame_log_name)
            
            self.frame_log_file = open(frame_log_path, 'w', newline='')
            self.frame_log_writer = csv.writer(self.frame_log_file)
            # Detailed Header for frame logs
            self.frame_log_writer.writerow(["frame_idx", "timestamp_sec", "ir_bbox", "ir_poly", "depth_m", "bbox_coords"])

            # Video Recorders
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.out_webcam = cv2.VideoWriter(file_web, fourcc, 20.0, (640, 480))
            
            self.rs_pipeline.stop()
            self.rs_config.enable_record_to_file(file_rs)
            self.rs_pipeline.start(self.rs_config)
            
            self.is_recording = True
            self.start_time_epoch = time.time()
            self.start_time_str = datetime.now()
            self.frame_count = 0
            
            self.btn_record.config(text="⬛ STOP (Saving Data...)", bg="black")
            self.lbl_status.config(text=f"Rec Frames: 0", fg="red")

        else:
            # === STOP ===
            self.rs_pipeline.stop()
            self.rs_config = rs.config()
            self.rs_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            self.rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            self.rs_config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
            self.rs_pipeline.start(self.rs_config)
            
            if self.out_webcam: self.out_webcam.release()
            
            # Close Frame Log
            if self.frame_log_file:
                self.frame_log_file.close()
                self.frame_log_file = None
                self.frame_log_writer = None

            duration = (datetime.now() - self.start_time_str).total_seconds()
            self.save_to_master_csv(duration)
            
            self.is_recording = False
            self.btn_record.config(text="🔴 REC", bg="red")
            self.lbl_status.config(text=f"Saved {self.frame_count} frames.", fg="green")

    def save_to_master_csv(self, duration):
        # Master log now points to the DETAILED CSV file
        new_row = [
            self.current_filename_base + "_Front.bag",
            self.current_filename_base + "_Top.mp4",
            self.current_frame_log_name, # <--- Link to detailed log
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            self.vars["Phase"].get(), self.vars["Bottle ID"].get(),
            self.vars["Content"].get(), self.vars["Volume"].get(), 
            self.vars["Note"].get(), self.vars["Plastic"].get(), 
            f"{duration:.2f}",
            str(self.saved_bbox_coords),
            str(self.saved_poly_coords),
            f"{self.val_bbox_ir:.2f}",
            f"{self.val_poly_ir:.2f}"
        ]
        
        with open(MASTER_CSV_FILE, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(new_row)

    def on_closing(self):
        try: self.rs_pipeline.stop()
        except: pass
        if self.cap_webcam.isOpened(): self.cap_webcam.release()
        if self.frame_log_file: self.frame_log_file.close()
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1200x700")
    app = TimeSeriesCollectorApp(root, "Time-Series Data Collector")
    root.mainloop()