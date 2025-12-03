import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import os
import csv
from datetime import datetime
from PIL import Image, ImageTk
import pandas as pd

# --- CONFIGURATION ---
DATASET_ROOT = "Oil_Quality_Dataset_2025"
RAW_DIR = os.path.join(DATASET_ROOT, "01_Raw_Recordings")
CSV_DIR = os.path.join(DATASET_ROOT, "02_Metadata_CSV")
CSV_FILE = os.path.join(CSV_DIR, "master_log.csv")

# ตรวจสอบและสร้างโฟลเดอร์
for folder in [RAW_DIR, CSV_DIR]:
    os.makedirs(folder, exist_ok=True)

# สร้างไฟล์ CSV หัวตารางถ้ายังไม่มี
if not os.path.exists(CSV_FILE):
    headers = ["filename", "timestamp", "phase", "bottle_id", "content_type", "note", "plastic_type", "duration_sec"]
    pd.DataFrame(columns=headers).to_csv(CSV_FILE, index=False)

class DataCollectorApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        
        # Camera Setup (0 คือ Webcam ปกติ, ถ้าเสียบ RealSense อาจจะเป็น 1 หรือ 2)
        self.video_source = 0 
        self.vid = cv2.VideoCapture(self.video_source)
        
        # ตั้งค่าขนาดภาพ (แนะนำ 640x480 หรือ 1280x720)
        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Variables for Recording
        self.is_recording = False
        self.out = None
        self.start_time = None
        self.current_filename = ""

        # --- UI LAYOUT ---
        # 1. Left Control Panel
        self.control_frame = tk.Frame(window, padx=10, pady=10)
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y)

        tk.Label(self.control_frame, text="Phase (การทดลอง):", font=("Arial", 10, "bold")).pack(anchor="w")
        self.phase_var = tk.StringVar(value="Phase1_Pure_Substances")
        self.cb_phase = ttk.Combobox(self.control_frame, textvariable=self.phase_var, state="readonly")
        self.cb_phase['values'] = ("Phase1_Pure_Substances", "Phase2_Mixtures")
        self.cb_phase.pack(fill=tk.X, pady=5)

        tk.Label(self.control_frame, text="Bottle ID (เลขขวด):", font=("Arial", 10, "bold")).pack(anchor="w")
        self.bottle_id_var = tk.StringVar(value="01")
        self.entry_bottle = tk.Entry(self.control_frame, textvariable=self.bottle_id_var)
        self.entry_bottle.pack(fill=tk.X, pady=5)

        tk.Label(self.control_frame, text="Content (ชนิดน้ำมัน):", font=("Arial", 10, "bold")).pack(anchor="w")
        self.content_var = tk.StringVar(value="Veg_Oil_Light")
        self.cb_content = ttk.Combobox(self.control_frame, textvariable=self.content_var)
        self.cb_content['values'] = ("Veg_Oil_Light", "Veg_Oil_Med", "Veg_Oil_Dark", 
                                     "Motor_Oil", "Lard_Oil", "Water", 
                                     "Mix_Veg_Motor", "Mix_Veg_Lard", "Mix_Veg_Water")
        self.cb_content.pack(fill=tk.X, pady=5)

        tk.Label(self.control_frame, text="Note (เช่น 10%, Hot):", font=("Arial", 10, "bold")).pack(anchor="w")
        self.note_var = tk.StringVar()
        self.entry_note = tk.Entry(self.control_frame, textvariable=self.note_var)
        self.entry_note.pack(fill=tk.X, pady=5)
        
        tk.Label(self.control_frame, text="Plastic Type (ชนิดขวด):", font=("Arial", 10, "bold")).pack(anchor="w")
        self.plastic_var = tk.StringVar(value="PP_Clear")
        self.cb_plastic = ttk.Combobox(self.control_frame, textvariable=self.plastic_var)
        self.cb_plastic['values'] = ("PP_Clear", "PP_Cloudy", "HDPE", "PET")
        self.cb_plastic.pack(fill=tk.X, pady=5)

        self.btn_record = tk.Button(self.control_frame, text="🔴 REC (Start)", bg="red", fg="white", 
                                    font=("Arial", 12, "bold"), command=self.toggle_recording)
        self.btn_record.pack(fill=tk.X, pady=20)
        
        self.lbl_status = tk.Label(self.control_frame, text="Ready", fg="green")
        self.lbl_status.pack()

        # 2. Right Video Preview
        self.canvas = tk.Canvas(window, width=640, height=480, bg="black")
        self.canvas.pack(side=tk.RIGHT)

        # Start Update Loop
        self.update()
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

    def toggle_recording(self):
        if not self.is_recording:
            # --- START RECORDING ---
            phase = self.phase_var.get()
            bottle = self.bottle_id_var.get()
            content = self.content_var.get()
            note = self.note_var.get().replace(" ", "_") # แทนที่เว้นวรรคด้วย _
            
            # สร้างชื่อไฟล์: Phase_B01_Content_Note_Time.mp4
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            note_str = f"_{note}" if note else ""
            filename = f"{phase}_B{bottle}_{content}{note_str}_{timestamp}.mp4"
            
            # สร้าง Path เก็บไฟล์ (แยก subfolder ตาม Phase)
            save_path = os.path.join(RAW_DIR, phase)
            os.makedirs(save_path, exist_ok=True)
            self.filepath = os.path.join(save_path, filename)
            self.current_filename = filename

            # Setup Video Writer (MP4)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
            self.out = cv2.VideoWriter(self.filepath, fourcc, 20.0, (640, 480))
            
            self.is_recording = True
            self.start_time = datetime.now()
            self.btn_record.config(text="⬛ STOP (Save)", bg="black")
            self.lbl_status.config(text=f"Recording: {filename}", fg="red")
            
        else:
            # --- STOP RECORDING ---
            self.is_recording = False
            self.out.release()
            self.out = None
            
            # Save CSV
            duration = (datetime.now() - self.start_time).total_seconds()
            self.save_to_csv(duration)
            
            self.btn_record.config(text="🔴 REC (Start)", bg="red")
            self.lbl_status.config(text=f"Saved: {self.current_filename}", fg="blue")
            messagebox.showinfo("Success", f"Video saved!\n{self.current_filename}")

    def save_to_csv(self, duration):
        new_row = [
            self.current_filename,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            self.phase_var.get(),
            self.bottle_id_var.get(),
            self.content_var.get(),
            self.note_var.get(),
            self.plastic_var.get(),
            f"{duration:.2f}"
        ]
        
        with open(CSV_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(new_row)

    def update(self):
        ret, frame = self.vid.read()
        if ret:
            # บันทึกวิดีโอถ้ากำลังอัดอยู่
            if self.is_recording and self.out is not None:
                self.out.write(frame)

            # แปลงภาพเพื่อแสดงบน GUI (OpenCV BGR -> RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(15, self.update) # เรียกฟังก์ชันนี้ซ้ำทุก 15ms

    def on_closing(self):
        if self.vid.isOpened():
            self.vid.release()
        if self.out is not None:
            self.out.release()
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = DataCollectorApp(root, "Oil Quality Data Collector (MP4 + CSV)")
    root.mainloop()