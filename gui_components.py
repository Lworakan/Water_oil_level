"""
GUI components for Oil Classification System
"""

import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import os

from config import *
from oil_bottle import extract_bottles_from_reference
from oil_detector import OilDetector


class OilClassifierGUI:
    """Main GUI application"""

    def __init__(self, root):
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.root.configure(bg='#2C3E50')

        # Initialize variables
        self.camera = None
        self.reference_img = None
        self.bottles = []
        self.detector = None
        self.matched_bottle = None
        self.is_running = False

        # Current measurements
        self.current_brightness = 0
        self.oil_volume_liters = 0.0
        self.oil_level_percent = 0.0
        self.binary_mask = None  # Binary preprocessed image

        # Load reference image and extract bottles
        self.load_reference_image()

        # Create detector
        if self.bottles:
            self.detector = OilDetector(self.bottles)

        # Create GUI layout
        self.create_gui()

        # Start camera
        self.start_camera()

        # Start update loop
        self.update_frame()

    def load_reference_image(self):
        """Load reference image and extract bottle data"""
        if not os.path.exists(REFERENCE_IMAGE):
            print(f"Error: Reference image '{REFERENCE_IMAGE}' not found")
            return

        self.reference_img = cv2.imread(REFERENCE_IMAGE)
        if self.reference_img is None:
            print("Error: Cannot load reference image")
            return

        # Extract bottles
        self.bottles = extract_bottles_from_reference(self.reference_img)

        if self.bottles:
            print(f"Loaded {len(self.bottles)} bottles from reference image")
            print(f"Brightness range: {self.bottles[-1].brightness:.1f} to {self.bottles[0].brightness:.1f}")

    def create_gui(self):
        """Create GUI layout"""
        # Title
        title_frame = tk.Frame(self.root, bg='#34495E', height=60)
        title_frame.pack(fill=tk.X, pady=(0, 10))

        title_label = tk.Label(title_frame, text=WINDOW_TITLE,
                              font=('Arial', 24, 'bold'), bg='#34495E', fg='white')
        title_label.pack(pady=10)

        # Main content frame
        content_frame = tk.Frame(self.root, bg='#2C3E50')
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Left side - Camera feed
        left_frame = tk.Frame(content_frame, bg='#34495E', relief=tk.RAISED, borderwidth=2)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        camera_label = tk.Label(left_frame, text="Live Camera Feed",
                               font=('Arial', 16, 'bold'), bg='#34495E', fg='white')
        camera_label.pack(pady=10)

        self.camera_canvas = tk.Canvas(left_frame, width=CAMERA_WIDTH, height=CAMERA_HEIGHT, bg='black')
        self.camera_canvas.pack(padx=10, pady=10)

        # Binary preprocessing display
        binary_label = tk.Label(left_frame, text="Binary Preprocessing (White=Oil, Black=Empty)",
                               font=('Arial', 14, 'bold'), bg='#34495E', fg='#F39C12')
        binary_label.pack(pady=(20, 5))

        self.binary_canvas = tk.Canvas(left_frame, width=320, height=240, bg='black')
        self.binary_canvas.pack(padx=10, pady=10)

        # Right side - Reference image and classification
        right_frame = tk.Frame(content_frame, bg='#34495E', relief=tk.RAISED, borderwidth=2)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        ref_label = tk.Label(right_frame, text="Reference Chart",
                            font=('Arial', 16, 'bold'), bg='#34495E', fg='white')
        ref_label.pack(pady=10)

        self.reference_canvas = tk.Canvas(right_frame, width=600, height=300, bg='black')
        self.reference_canvas.pack(padx=10, pady=10)

        # Classification info panel
        self.create_info_panel(right_frame)

        # Control buttons
        self.create_control_buttons()

    def create_info_panel(self, parent):
        """Create the information display panel"""
        info_frame = tk.Frame(parent, bg='#2C3E50', relief=tk.RIDGE, borderwidth=3)
        info_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Status label
        self.status_label = tk.Label(info_frame, text="STATUS: READY",
                                     font=('Arial', 20, 'bold'),
                                     bg='#3498DB', fg='white',
                                     relief=tk.RAISED, borderwidth=3)
        self.status_label.pack(fill=tk.X, padx=20, pady=(20, 10))

        # Brightness value label
        self.value_label = tk.Label(info_frame, text="BRIGHTNESS: -",
                                   font=('Arial', 18, 'bold'),
                                   bg='#7F8C8D', fg='white',
                                   relief=tk.RAISED, borderwidth=3)
        self.value_label.pack(fill=tk.X, padx=20, pady=10)

        # Gradient bar
        self.create_gradient_scale(info_frame)

        # Confidence label
        self.confidence_label = tk.Label(info_frame, text="Confidence: -",
                                        font=('Arial', 14),
                                        bg='#2C3E50', fg='white')
        self.confidence_label.pack(pady=10)

        # Oil Volume Display
        self.create_volume_display(info_frame)

    def create_gradient_scale(self, parent):
        """Create gradient scale display"""
        scale_frame = tk.Frame(parent, bg='#2C3E50')
        scale_frame.pack(pady=20, padx=20)

        scale_title = tk.Label(scale_frame, text="Oil Darkness Scale (0-255)",
                              font=('Arial', 12, 'bold'), bg='#2C3E50', fg='white')
        scale_title.pack(pady=(0, 5))

        # Canvas for gradient bar
        self.gradient_canvas = tk.Canvas(scale_frame, width=520, height=120,
                                        bg='#2C3E50', highlightthickness=0)
        self.gradient_canvas.pack()

        # Scale labels
        scale_labels_frame = tk.Frame(scale_frame, bg='#2C3E50')
        scale_labels_frame.pack(pady=(5, 0))

        tk.Label(scale_labels_frame, text="255\n(Bright/Light)",
                font=('Arial', 9), bg='#2C3E50', fg='white').pack(side=tk.LEFT, padx=(0, 150))
        tk.Label(scale_labels_frame, text="128\n(Medium)",
                font=('Arial', 9), bg='#2C3E50', fg='white').pack(side=tk.LEFT, padx=100)
        tk.Label(scale_labels_frame, text="0\n(Dark/Black)",
                font=('Arial', 9), bg='#2C3E50', fg='white').pack(side=tk.LEFT, padx=(150, 0))

    def create_volume_display(self, parent):
        """Create volume estimation display"""
        volume_frame = tk.Frame(parent, bg='#34495E', relief=tk.RIDGE, borderwidth=3)
        volume_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        volume_title = tk.Label(volume_frame, text="OIL VOLUME ESTIMATION",
                               font=('Arial', 14, 'bold'), bg='#34495E', fg='white')
        volume_title.pack(pady=(10, 5))

        # Volume display
        self.volume_label = tk.Label(volume_frame, text="0.00 L",
                                    font=('Arial', 32, 'bold'),
                                    bg='#34495E', fg='#3498DB')
        self.volume_label.pack(pady=10)

        # Percentage display
        self.percent_label = tk.Label(volume_frame, text="0%",
                                     font=('Arial', 18, 'bold'),
                                     bg='#34495E', fg='#ECF0F1')
        self.percent_label.pack(pady=5)

        # Visual volume gauge
        gauge_container = tk.Frame(volume_frame, bg='#34495E')
        gauge_container.pack(pady=10)

        tk.Label(gauge_container, text="Fill Level:",
                font=('Arial', 10), bg='#34495E', fg='white').pack(side=tk.LEFT, padx=(0, 10))

        self.volume_gauge_canvas = tk.Canvas(gauge_container, width=200, height=30,
                                            bg='#2C3E50', highlightthickness=1,
                                            highlightbackground='#7F8C8D')
        self.volume_gauge_canvas.pack(side=tk.LEFT)

        # Info text
        info_text = tk.Label(volume_frame,
                           text=f"Container Capacity: {CONTAINER_CAPACITY_LITERS} Liter\nWhite = Empty | Colors = Oil",
                           font=('Arial', 9), bg='#34495E', fg='#BDC3C7')
        info_text.pack(pady=(10, 15))

    def create_control_buttons(self):
        """Create control buttons"""
        button_frame = tk.Frame(self.root, bg='#2C3E50')
        button_frame.pack(pady=10)

        self.toggle_btn = tk.Button(button_frame, text="Stop",
                                    command=self.toggle_camera,
                                    font=('Arial', 12, 'bold'),
                                    bg='#E74C3C', fg='white',
                                    width=15, height=2)
        self.toggle_btn.pack(side=tk.LEFT, padx=10)

        quit_btn = tk.Button(button_frame, text="Quit",
                           command=self.quit_app,
                           font=('Arial', 12, 'bold'),
                           bg='#C0392B', fg='white',
                           width=15, height=2)
        quit_btn.pack(side=tk.LEFT, padx=10)

    def start_camera(self):
        """Initialize camera"""
        self.camera = cv2.VideoCapture(CAMERA_INDEX)
        if not self.camera.isOpened():
            print(f"Error: Cannot open camera {CAMERA_INDEX}")
            return

        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.is_running = True

    def toggle_camera(self):
        """Toggle camera on/off"""
        self.is_running = not self.is_running
        if self.is_running:
            self.toggle_btn.config(text="Stop", bg='#E74C3C')
        else:
            self.toggle_btn.config(text="Start", bg='#27AE60')

    def draw_gradient_bar_with_indicator(self):
        """Draw gradient bar with current value indicator"""
        if self.detector is None or self.detector.gradient_bar is None:
            return

        # Create display version
        display_bar = self.detector.gradient_bar.copy()
        bar_width = display_bar.shape[1]
        bar_height = display_bar.shape[0]

        # Calculate indicator position
        indicator_x = int((255 - self.current_brightness) / 255 * bar_width)
        indicator_x = max(0, min(bar_width - 1, indicator_x))

        # Draw indicator line
        cv2.line(display_bar, (indicator_x, 0), (indicator_x, bar_height),
                (0, 0, 255), 4)

        # Draw triangle marker
        triangle_pts = np.array([
            [indicator_x, 0],
            [indicator_x - 10, -15],
            [indicator_x + 10, -15]
        ], np.int32)

        temp_img = np.zeros((bar_height + 20, bar_width, 3), dtype=np.uint8)
        temp_img[:] = (44, 62, 80)
        temp_img[20:, :] = display_bar

        cv2.fillPoly(temp_img, [triangle_pts + np.array([0, 20])], (0, 0, 255))
        cv2.polylines(temp_img, [triangle_pts + np.array([0, 20])], True, (255, 255, 255), 2)

        # Add value text
        value_text = f"{int(self.current_brightness)}"
        text_size = cv2.getTextSize(value_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        text_x = indicator_x - text_size[0] // 2
        cv2.putText(temp_img, value_text, (text_x, 13),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Convert and display
        display_rgb = cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(display_rgb)
        photo = ImageTk.PhotoImage(image=img)
        self.gradient_canvas.create_image(4, 0, image=photo, anchor=tk.NW)
        self.gradient_canvas.image = photo

    def draw_volume_gauge(self):
        """Draw volume gauge bar"""
        self.volume_gauge_canvas.delete("all")

        # Draw background
        self.volume_gauge_canvas.create_rectangle(0, 0, 200, 30,
                                                  fill='#2C3E50', outline='#7F8C8D')

        # Calculate fill width
        fill_width = int((self.oil_level_percent / 100.0) * 198)

        # Color based on fill level
        if self.oil_level_percent < 25:
            fill_color = '#E74C3C'  # Red
        elif self.oil_level_percent < 50:
            fill_color = '#F39C12'  # Orange
        elif self.oil_level_percent < 75:
            fill_color = '#F1C40F'  # Yellow
        else:
            fill_color = '#27AE60'  # Green

        # Draw fill bar
        if fill_width > 0:
            self.volume_gauge_canvas.create_rectangle(1, 1, fill_width + 1, 29,
                                                     fill=fill_color, outline='')

        # Draw percentage text
        self.volume_gauge_canvas.create_text(100, 15,
                                            text=f"{self.oil_level_percent:.1f}%",
                                            font=('Arial', 12, 'bold'),
                                            fill='white')

    def update_frame(self):
        """Update camera feed and classification"""
        if self.camera and self.camera.isOpened() and self.is_running and self.detector:
            ret, frame = self.camera.read()
            if ret:
                # Draw ROI and oil level
                self.draw_roi_and_level(frame)

                # Analyze oil color
                camera_color = self.detector.analyze_sample(frame)
                self.current_brightness = camera_color['hsv'][2]

                # Detect oil volume (now returns binary mask too)
                self.oil_volume_liters, self.oil_level_percent, oil_px, total_px, self.binary_mask = \
                    self.detector.detect_oil_volume(frame)

                # Update volume displays
                self.volume_label.config(text=f"{self.oil_volume_liters:.2f} L")
                self.percent_label.config(text=f"{self.oil_level_percent:.1f}%")

                # Check if water
                is_water = self.detector.is_water(camera_color)

                if is_water:
                    self.update_water_status()
                else:
                    self.update_oil_status(camera_color)

                # Display camera feed
                self.display_camera_frame(frame)

        # Update visualizations
        self.draw_gradient_bar_with_indicator()
        self.draw_volume_gauge()
        self.display_binary_mask()
        self.update_reference_display()

        # Schedule next update
        self.root.after(FRAME_UPDATE_MS, self.update_frame)

    def draw_roi_and_level(self, frame):
        """Draw ROI box and oil level line on frame"""
        h, w = frame.shape[:2]
        x1, y1 = w // 4, h // 4
        x2, y2 = 3 * w // 4, 3 * h // 4

        # Draw ROI
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(frame, "Oil Detection Area", (x1, y1 - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Draw oil level line
        sample_h = y2 - y1
        oil_level_y = y2 - int((self.oil_level_percent / 100.0) * sample_h)
        if self.oil_level_percent > 5:
            cv2.line(frame, (x1, oil_level_y), (x2, oil_level_y), (0, 255, 255), 2)
            cv2.putText(frame, f"{self.oil_level_percent:.0f}%", (x2 + 10, oil_level_y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    def update_water_status(self):
        """Update UI for water detection"""
        self.status_label.config(text="STATUS: WATER / NO OIL",
                               bg='#3498DB', fg='white')
        self.value_label.config(text=f"BRIGHTNESS: {int(self.current_brightness)} (WATER)",
                              bg='#FFFFFF', fg='#000000')
        self.confidence_label.config(text="Detection: Water or No Oil Present")
        self.matched_bottle = None

    def update_oil_status(self, camera_color):
        """Update UI for oil detection"""
        self.matched_bottle, confidence = self.detector.find_matching_bottle(camera_color)

        if self.matched_bottle:
            # Update status
            if self.matched_bottle.is_usable:
                self.status_label.config(text="STATUS: USABLE OIL ✓",
                                       bg='#27AE60', fg='white')
            else:
                self.status_label.config(text="STATUS: UNUSABLE OIL ✗",
                                       bg='#E74C3C', fg='white')

            # Update brightness
            darkness_percent = (255 - self.current_brightness) / 255 * 100
            self.value_label.config(text=f"BRIGHTNESS: {int(self.current_brightness)} / 255",
                                  bg='#7F8C8D', fg='white')

            # Update confidence
            self.confidence_label.config(
                text=f"Confidence: {confidence:.1f}%  |  Darkness: {darkness_percent:.1f}%"
            )

    def display_camera_frame(self, frame):
        """Display camera frame on canvas"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img = img.resize((CAMERA_WIDTH, CAMERA_HEIGHT), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image=img)
        self.camera_canvas.create_image(0, 0, image=photo, anchor=tk.NW)
        self.camera_canvas.image = photo

    def display_binary_mask(self):
        """Display binary preprocessing mask"""
        if self.binary_mask is None:
            return

        # Convert to RGB for display
        binary_rgb = cv2.cvtColor(self.binary_mask, cv2.COLOR_BGR2RGB)

        # Add text overlay showing pixel counts
        h, w = binary_rgb.shape[:2]
        oil_pixels = np.sum(self.binary_mask[:, :, 0] == 255)
        total_pixels = h * w

        # Add info text
        info_text = f"Oil: {oil_pixels} px | Empty: {total_pixels - oil_pixels} px"
        cv2.putText(binary_rgb, info_text, (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Resize and display
        img = Image.fromarray(binary_rgb)
        img = img.resize((320, 240), Image.Resampling.NEAREST)  # Use NEAREST for binary
        photo = ImageTk.PhotoImage(image=img)
        self.binary_canvas.create_image(0, 0, image=photo, anchor=tk.NW)
        self.binary_canvas.image = photo

    def update_reference_display(self):
        """Update reference image with highlighted matching bottle"""
        if self.reference_img is None:
            return

        display_img = self.reference_img.copy()

        # Draw bounding box on matched bottle
        if self.matched_bottle:
            bottle = self.matched_bottle
            x, y, w, h = bottle.x, bottle.y, bottle.w, bottle.h

            color = (0, 255, 0) if bottle.is_usable else (0, 0, 255)
            cv2.rectangle(display_img, (x, y), (x + w, y + h), color, 6)

            # Add label
            label = f"V:{int(bottle.brightness)}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.rectangle(display_img, (x, y - 35), (x + label_size[0] + 10, y), color, -1)
            cv2.putText(display_img, label, (x + 5, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Convert and display
        display_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(display_rgb)
        img = img.resize((600, 300), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image=img)
        self.reference_canvas.create_image(0, 0, image=photo, anchor=tk.NW)
        self.reference_canvas.image = photo

    def quit_app(self):
        """Clean up and quit"""
        if self.camera:
            self.camera.release()
        self.root.quit()
