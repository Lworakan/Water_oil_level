#!/usr/bin/env python3
"""
Oil Classification GUI System
Displays reference chart with gradient scale bar (0-255) and highlights matching bottle
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os

# Configuration
REFERENCE_IMAGE = "unnamed.jpg"
CAMERA_INDEX = 0

# White threshold for water/no oil detection
WHITE_THRESHOLD = 200  # If brightness > 200, it's likely water or no oil

# Container capacity
CONTAINER_CAPACITY_LITERS = 1.0  # Bottom container holds 1 liter

class OilBottle:
    """Represents a bottle in the reference image"""
    def __init__(self, x, y, w, h, color_data, is_usable, bottle_index):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.color_data = color_data  # {'hsv': (h,s,v), 'lab': (l,a,b), 'bgr': (b,g,r)}
        self.is_usable = is_usable
        self.bottle_index = bottle_index

class OilClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Oil Classification System - Volume & Quality Detection")
        self.root.geometry("1500x950")
        self.root.configure(bg='#2C3E50')

        # Initialize variables
        self.camera = None
        self.reference_img = None
        self.bottles = []  # List of OilBottle objects sorted by brightness
        self.matched_bottle = None
        self.is_running = False
        self.current_value = 0  # Current brightness value (0-255)
        self.gradient_bar = None  # Gradient bar image

        # Oil volume tracking
        self.oil_volume_liters = 0.0  # Current estimated volume
        self.oil_level_percent = 0.0  # Fill level percentage (0-100%)

        # Load reference image and extract bottles
        self.load_reference_image()

        # Create gradient bar from bottle colors
        self.create_gradient_bar()

        # Create GUI layout
        self.create_gui()

        # Start camera
        self.start_camera()

        # Start update loop
        self.update_frame()

    def load_reference_image(self):
        """Load reference image and detect bottle positions"""
        if not os.path.exists(REFERENCE_IMAGE):
            print(f"Error: Reference image '{REFERENCE_IMAGE}' not found")
            return

        self.reference_img = cv2.imread(REFERENCE_IMAGE)
        if self.reference_img is None:
            print("Error: Cannot load reference image")
            return

        h, w = self.reference_img.shape[:2]

        # Define bottle grid (2 rows x 10 columns)
        num_bottles_per_row = 10

        # Top row (usable oil) - y range
        top_row_y1 = int(h * 0.15)
        top_row_y2 = int(h * 0.45)

        # Bottom row (unusable oil) - y range
        bottom_row_y1 = int(h * 0.55)
        bottom_row_y2 = int(h * 0.85)

        bottle_width = w // num_bottles_per_row

        all_bottles = []

        # Extract top row bottles (USABLE)
        for i in range(num_bottles_per_row):
            x1 = i * bottle_width + int(bottle_width * 0.1)
            x2 = (i + 1) * bottle_width - int(bottle_width * 0.1)
            y1 = top_row_y1 + int((top_row_y2 - top_row_y1) * 0.2)
            y2 = top_row_y2 - int((top_row_y2 - top_row_y1) * 0.1)

            color_data = self.extract_bottle_color(x1, y1, x2, y2)
            bottle = OilBottle(x1, y1, x2 - x1, y2 - y1, color_data,
                             is_usable=True, bottle_index=i)
            all_bottles.append(bottle)

        # Extract bottom row bottles (UNUSABLE)
        for i in range(num_bottles_per_row):
            x1 = i * bottle_width + int(bottle_width * 0.1)
            x2 = (i + 1) * bottle_width - int(bottle_width * 0.1)
            y1 = bottom_row_y1 + int((bottom_row_y2 - bottom_row_y1) * 0.2)
            y2 = bottom_row_y2 - int((bottom_row_y2 - bottom_row_y1) * 0.1)

            color_data = self.extract_bottle_color(x1, y1, x2, y2)
            bottle = OilBottle(x1, y1, x2 - x1, y2 - y1, color_data,
                             is_usable=False, bottle_index=i + num_bottles_per_row)
            all_bottles.append(bottle)

        # Sort bottles by brightness (descending - brightest first)
        self.bottles = sorted(all_bottles, key=lambda b: b.color_data['hsv'][2], reverse=True)

        print(f"Loaded {len(self.bottles)} bottles from reference image")
        print(f"Brightness range: {self.bottles[-1].color_data['hsv'][2]:.1f} to {self.bottles[0].color_data['hsv'][2]:.1f}")

    def extract_bottle_color(self, x1, y1, x2, y2):
        """Extract average color from bottle region"""
        sample = self.reference_img[y1:y2, x1:x2]

        # Convert to HSV and LAB
        hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(sample, cv2.COLOR_BGR2LAB)

        # Calculate average color values
        avg_hue = np.mean(hsv[:, :, 0])
        avg_sat = np.mean(hsv[:, :, 1])
        avg_val = np.mean(hsv[:, :, 2])
        avg_l = np.mean(lab[:, :, 0])
        avg_a = np.mean(lab[:, :, 1])
        avg_b = np.mean(lab[:, :, 2])

        # Get average BGR color
        avg_bgr = np.mean(sample, axis=(0, 1))

        return {
            'hsv': (avg_hue, avg_sat, avg_val),
            'lab': (avg_l, avg_a, avg_b),
            'bgr': tuple(avg_bgr)
        }

    def create_gradient_bar(self):
        """Create gradient bar from bottle colors (0-255 scale)"""
        if not self.bottles:
            return

        # Create gradient image (width=512, height=80)
        bar_width = 512
        bar_height = 80
        gradient = np.zeros((bar_height, bar_width, 3), dtype=np.uint8)

        # Map brightness values to positions
        # Get min and max brightness from bottles
        min_brightness = min(b.color_data['hsv'][2] for b in self.bottles)
        max_brightness = max(b.color_data['hsv'][2] for b in self.bottles)

        # Create smooth gradient by interpolating between bottle colors
        for x in range(bar_width):
            # Map x position to brightness value (0-255)
            brightness = 255 - (x / bar_width * 255)  # Reverse: left=bright, right=dark

            # Find two nearest bottles for interpolation
            closest_bottles = sorted(self.bottles,
                                    key=lambda b: abs(b.color_data['hsv'][2] - brightness))[:2]

            if len(closest_bottles) >= 2:
                b1 = closest_bottles[0]
                b2 = closest_bottles[1]

                # Calculate interpolation weight
                v1 = b1.color_data['hsv'][2]
                v2 = b2.color_data['hsv'][2]

                if abs(v1 - v2) > 0.1:
                    weight = abs(brightness - v2) / abs(v1 - v2)
                    weight = np.clip(weight, 0, 1)
                else:
                    weight = 0.5

                # Interpolate BGR color
                color = (
                    int(b1.color_data['bgr'][0] * weight + b2.color_data['bgr'][0] * (1 - weight)),
                    int(b1.color_data['bgr'][1] * weight + b2.color_data['bgr'][1] * (1 - weight)),
                    int(b1.color_data['bgr'][2] * weight + b2.color_data['bgr'][2] * (1 - weight))
                )
            else:
                # Use closest bottle color
                color = tuple(map(int, closest_bottles[0].color_data['bgr']))

            # Fill column
            gradient[:, x] = color

        self.gradient_bar = gradient

    def create_gui(self):
        """Create GUI layout"""
        # Title
        title_frame = tk.Frame(self.root, bg='#34495E', height=60)
        title_frame.pack(fill=tk.X, pady=(0, 10))

        title_label = tk.Label(title_frame, text="Oil Classification & Volume Estimation System",
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

        self.camera_canvas = tk.Canvas(left_frame, width=640, height=480, bg='black')
        self.camera_canvas.pack(padx=10, pady=10)

        # Right side - Reference image and classification
        right_frame = tk.Frame(content_frame, bg='#34495E', relief=tk.RAISED, borderwidth=2)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        ref_label = tk.Label(right_frame, text="Reference Chart",
                            font=('Arial', 16, 'bold'), bg='#34495E', fg='white')
        ref_label.pack(pady=10)

        self.reference_canvas = tk.Canvas(right_frame, width=600, height=300, bg='black')
        self.reference_canvas.pack(padx=10, pady=10)

        # Classification info panel
        info_frame = tk.Frame(right_frame, bg='#2C3E50', relief=tk.RIDGE, borderwidth=3)
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

        # Gradient bar with scale
        scale_frame = tk.Frame(info_frame, bg='#2C3E50')
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

        # Confidence label
        self.confidence_label = tk.Label(info_frame, text="Confidence: -",
                                        font=('Arial', 14),
                                        bg='#2C3E50', fg='white')
        self.confidence_label.pack(pady=10)

        # Oil Volume Display Section
        volume_frame = tk.Frame(info_frame, bg='#34495E', relief=tk.RIDGE, borderwidth=3)
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

        # Visual volume gauge (vertical bar)
        gauge_container = tk.Frame(volume_frame, bg='#34495E')
        gauge_container.pack(pady=10)

        tk.Label(gauge_container, text="Fill Level:",
                font=('Arial', 10), bg='#34495E', fg='white').pack(side=tk.LEFT, padx=(0, 10))

        self.volume_gauge_canvas = tk.Canvas(gauge_container, width=200, height=30,
                                            bg='#2C3E50', highlightthickness=1,
                                            highlightbackground='#7F8C8D')
        self.volume_gauge_canvas.pack(side=tk.LEFT)

        # Capacity label
        capacity_label = tk.Label(volume_frame,
                                 text=f"Container Capacity: {CONTAINER_CAPACITY_LITERS} Liter",
                                 font=('Arial', 10), bg='#34495E', fg='#BDC3C7')
        capacity_label.pack(pady=(10, 15))

        # Control buttons
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

        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.is_running = True

    def toggle_camera(self):
        """Toggle camera on/off"""
        self.is_running = not self.is_running
        if self.is_running:
            self.toggle_btn.config(text="Stop", bg='#E74C3C')
        else:
            self.toggle_btn.config(text="Start", bg='#27AE60')

    def analyze_oil_sample(self, frame):
        """Analyze oil from camera frame (center region)"""
        h, w = frame.shape[:2]
        x1, y1 = w // 4, h // 4
        x2, y2 = 3 * w // 4, 3 * h // 4

        sample = frame[y1:y2, x1:x2]

        # Convert to HSV and LAB
        hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(sample, cv2.COLOR_BGR2LAB)

        # Calculate average color values
        avg_hue = np.mean(hsv[:, :, 0])
        avg_sat = np.mean(hsv[:, :, 1])
        avg_val = np.mean(hsv[:, :, 2])
        avg_l = np.mean(lab[:, :, 0])
        avg_a = np.mean(lab[:, :, 1])
        avg_b = np.mean(lab[:, :, 2])

        # Get average BGR
        avg_bgr = np.mean(sample, axis=(0, 1))

        return {
            'hsv': (avg_hue, avg_sat, avg_val),
            'lab': (avg_l, avg_a, avg_b),
            'bgr': tuple(avg_bgr)
        }

    def color_distance(self, color1, color2):
        """Calculate color distance"""
        # HSV distance (weighted)
        h1, s1, v1 = color1['hsv']
        h2, s2, v2 = color2['hsv']
        hsv_dist = np.sqrt((h2 - h1)**2 * 0.5 + (s2 - s1)**2 * 0.3 + (v2 - v1)**2 * 1.0)

        # LAB distance
        l1, a1, b1 = color1['lab']
        l2, a2, b2 = color2['lab']
        lab_dist = np.sqrt((l2 - l1)**2 + (a2 - a1)**2 + (b2 - b1)**2)

        return hsv_dist * 0.3 + lab_dist * 0.7

    def detect_oil_level(self, frame):
        """
        Detect oil level in the container by analyzing the frame
        Returns: (volume_liters, level_percent)
        """
        h, w = frame.shape[:2]
        x1, y1 = w // 4, h // 4
        x2, y2 = 3 * w // 4, 3 * h // 4

        # Use the sample region
        sample = frame[y1:y2, x1:x2]
        sample_h, sample_w = sample.shape[:2]

        # Convert to grayscale for level detection
        gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)

        # Threshold to detect oil vs empty space
        # Oil is darker than empty/air, so we threshold
        # Assuming bottom is filled from bottom up
        _, binary = cv2.threshold(gray, WHITE_THRESHOLD - 20, 255, cv2.THRESH_BINARY_INV)

        # Calculate percentage of oil pixels (darker areas)
        oil_pixels = np.sum(binary > 0)
        total_pixels = binary.size
        fill_ratio = oil_pixels / total_pixels

        # Alternative method: scan from bottom to top to find oil level
        # Check each row from bottom to find the transition point
        row_darkness = np.mean(gray, axis=1)  # Average darkness per row

        # Find the highest row with significant darkness (oil present)
        oil_threshold = WHITE_THRESHOLD - 30
        oil_rows = np.where(row_darkness < oil_threshold)[0]

        if len(oil_rows) > 0:
            # Oil level is from bottom (sample_h) to highest oil row
            highest_oil_row = oil_rows[0]  # Top-most row with oil
            level_from_bottom = sample_h - highest_oil_row
            level_percent = (level_from_bottom / sample_h) * 100
        else:
            # No oil detected or very little
            level_percent = fill_ratio * 100

        # Clamp between 0-100%
        level_percent = max(0, min(100, level_percent))

        # Calculate volume in liters
        volume_liters = (level_percent / 100.0) * CONTAINER_CAPACITY_LITERS

        return volume_liters, level_percent

    def find_matching_bottle(self, camera_color):
        """Find the closest matching bottle"""
        if not self.bottles:
            return None, 0

        min_distance = float('inf')
        matched = None

        for bottle in self.bottles:
            dist = self.color_distance(camera_color, bottle.color_data)
            if dist < min_distance:
                min_distance = dist
                matched = bottle

        # Calculate confidence (inverse of normalized distance)
        max_dist = 150  # Typical max distance
        confidence = max(0, min(100, (1 - min_distance / max_dist) * 100))

        return matched, confidence

    def draw_gradient_bar_with_indicator(self):
        """Draw gradient bar with current value indicator"""
        if self.gradient_bar is None:
            return

        # Create display version
        display_bar = self.gradient_bar.copy()
        bar_width = display_bar.shape[1]
        bar_height = display_bar.shape[0]

        # Calculate indicator position (0-255 mapped to bar width)
        # Left = 255 (bright), Right = 0 (dark)
        indicator_x = int((255 - self.current_value) / 255 * bar_width)
        indicator_x = max(0, min(bar_width - 1, indicator_x))

        # Draw indicator line (red vertical line)
        cv2.line(display_bar, (indicator_x, 0), (indicator_x, bar_height),
                (0, 0, 255), 4)

        # Draw triangle marker at top
        triangle_pts = np.array([
            [indicator_x, 0],
            [indicator_x - 10, -15],
            [indicator_x + 10, -15]
        ], np.int32)

        # Add border to make marker stand out
        temp_img = np.zeros((bar_height + 20, bar_width, 3), dtype=np.uint8)
        temp_img[:] = (44, 62, 80)  # Match background color
        temp_img[20:, :] = display_bar

        # Draw triangle
        cv2.fillPoly(temp_img, [triangle_pts + np.array([0, 20])], (0, 0, 255))
        cv2.polylines(temp_img, [triangle_pts + np.array([0, 20])], True, (255, 255, 255), 2)

        # Add value text on triangle
        value_text = f"{int(self.current_value)}"
        text_size = cv2.getTextSize(value_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        text_x = indicator_x - text_size[0] // 2
        text_y = 13
        cv2.putText(temp_img, value_text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Convert and display
        display_rgb = cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(display_rgb)
        photo = ImageTk.PhotoImage(image=img)
        self.gradient_canvas.create_image(4, 0, image=photo, anchor=tk.NW)
        self.gradient_canvas.image = photo

    def draw_volume_gauge(self):
        """Draw volume gauge bar"""
        # Clear canvas
        self.volume_gauge_canvas.delete("all")

        # Draw background
        self.volume_gauge_canvas.create_rectangle(0, 0, 200, 30,
                                                  fill='#2C3E50', outline='#7F8C8D')

        # Calculate fill width based on percentage
        fill_width = int((self.oil_level_percent / 100.0) * 198)

        # Color based on fill level
        if self.oil_level_percent < 25:
            fill_color = '#E74C3C'  # Red - low
        elif self.oil_level_percent < 50:
            fill_color = '#F39C12'  # Orange - medium-low
        elif self.oil_level_percent < 75:
            fill_color = '#F1C40F'  # Yellow - medium
        else:
            fill_color = '#27AE60'  # Green - high

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
        if self.camera and self.camera.isOpened() and self.is_running:
            ret, frame = self.camera.read()
            if ret:
                # Draw ROI
                h, w = frame.shape[:2]
                x1, y1 = w // 4, h // 4
                x2, y2 = 3 * w // 4, 3 * h // 4
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(frame, "Oil Sample Area", (x1, y1 - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # Draw oil level line
                sample_h = y2 - y1
                oil_level_y = y2 - int((self.oil_level_percent / 100.0) * sample_h)
                if self.oil_level_percent > 5:  # Only show if there's significant oil
                    cv2.line(frame, (x1, oil_level_y), (x2, oil_level_y), (0, 255, 255), 2)
                    cv2.putText(frame, f"{self.oil_level_percent:.0f}%", (x2 + 10, oil_level_y + 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # Analyze oil
                camera_color = self.analyze_oil_sample(frame)
                self.current_value = camera_color['hsv'][2]  # Brightness value

                # Detect oil volume
                self.oil_volume_liters, self.oil_level_percent = self.detect_oil_level(frame)

                # Update volume displays
                self.volume_label.config(text=f"{self.oil_volume_liters:.2f} L")
                self.percent_label.config(text=f"{self.oil_level_percent:.1f}%")

                # Check if it's water (white/very bright)
                is_water = (self.current_value >= WHITE_THRESHOLD and
                           camera_color['hsv'][1] < 30)  # Low saturation

                if is_water:
                    # Water detected
                    self.status_label.config(text="STATUS: WATER / NO OIL",
                                           bg='#3498DB', fg='white')
                    self.value_label.config(text=f"BRIGHTNESS: {int(self.current_value)} (WATER)",
                                          bg='#FFFFFF', fg='#000000')
                    self.confidence_label.config(text="Detection: Water or No Oil Present")
                    self.matched_bottle = None
                else:
                    # Oil detected - find matching bottle
                    self.matched_bottle, confidence = self.find_matching_bottle(camera_color)

                    if self.matched_bottle:
                        # Update status
                        if self.matched_bottle.is_usable:
                            self.status_label.config(text="STATUS: USABLE OIL ✓",
                                                   bg='#27AE60', fg='white')
                        else:
                            self.status_label.config(text="STATUS: UNUSABLE OIL ✗",
                                                   bg='#E74C3C', fg='white')

                        # Update brightness value display
                        darkness_percent = (255 - self.current_value) / 255 * 100
                        self.value_label.config(text=f"BRIGHTNESS: {int(self.current_value)} / 255",
                                              bg='#7F8C8D', fg='white')

                        # Update confidence
                        self.confidence_label.config(text=f"Confidence: {confidence:.1f}%  |  Darkness: {darkness_percent:.1f}%")

                # Display camera feed
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img = img.resize((640, 480), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(image=img)
                self.camera_canvas.create_image(0, 0, image=photo, anchor=tk.NW)
                self.camera_canvas.image = photo

        # Update gradient bar with indicator
        self.draw_gradient_bar_with_indicator()

        # Update volume gauge
        self.draw_volume_gauge()

        # Update reference image with bounding box
        self.update_reference_display()

        # Schedule next update
        self.root.after(30, self.update_frame)

    def update_reference_display(self):
        """Update reference image with highlighted matching bottle"""
        if self.reference_img is None:
            return

        # Create copy of reference image
        display_img = self.reference_img.copy()

        # Draw bounding box on matched bottle
        if self.matched_bottle:
            bottle = self.matched_bottle
            x, y, w, h = bottle.x, bottle.y, bottle.w, bottle.h

            # Draw thick green box for usable, red for unusable
            color = (0, 255, 0) if bottle.is_usable else (0, 0, 255)
            cv2.rectangle(display_img, (x, y), (x + w, y + h), color, 6)

            # Add label with brightness value
            label = f"V:{int(bottle.color_data['hsv'][2])}"
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

def main():
    root = tk.Tk()
    app = OilClassifierGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
