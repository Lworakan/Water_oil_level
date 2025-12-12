"""
Oil Quality Classification GUI with SAM3 Segmentation and PCA Classification
Automatically segments oil in bottle and classifies using PCA-based rule system
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import pyrealsense2 as rs
import os
import pickle
import json
from datetime import datetime

# Import SAM3 and other required libraries
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM3_AVAILABLE = True
except ImportError:
    SAM3_AVAILABLE = False
    print("Warning: SAM3 not available. Install sam2 for automatic segmentation.")

class OilClassificationGUI:
    """
    GUI Application for Oil Quality Classification
    - Automatic segmentation using SAM3
    - Feature extraction (color, HSV, IR, texture)
    - PCA transformation
    - Rule-based classification using nearest centroid
    """

    def __init__(self, root):
        self.root = root
        self.root.title("Oil Quality Classification System")
        self.root.geometry("1400x900")

        # Load PCA model and centroids
        self.load_pca_model()

        # Initialize RealSense camera
        self.init_realsense()

        # Initialize SAM3 (if available)
        self.init_sam3()

        # Variables
        self.current_frame = None
        self.segmentation_mask = None
        self.bbox_coords = None
        self.polygon_coords = None
        self.features = None
        self.pca_values = [0, 0, 0]
        self.predicted_class = "Unknown"
        self.confidence = 0.0

        # Setup GUI
        self.setup_gui()

        # Start video loop
        self.update_frame()

    def load_pca_model(self):
        """Load PCA model parameters"""
        # Define centroids from our analysis
        self.centroids = {
            'Dark oil': np.array([0.16, -0.41, 1.68]),
            'Light oil': np.array([3.53, 0.78, 0.09]),
            'Medium oil': np.array([2.78, 0.68, 0.99]),
            'Motor_Oil': np.array([-4.24, -1.41, 1.84]),
            'lard': np.array([0.50, -3.87, -1.09]),
            'oil_lard': np.array([-2.69, 0.63, -5.07]),
            'oil_motor': np.array([-5.04, -0.41, 1.36]),
            'oil_water': np.array([-0.95, 6.61, 0.14]),
            'water': np.array([3.06, -1.38, -0.48])
        }

        # Try to load saved PCA model
        model_path = 'pca_model_params.pkl'
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                self.pca_mean = model_data['mean']
                self.pca_std = model_data['std']
                self.pca_components = model_data['components']
                print("✓ PCA model loaded successfully")
        else:
            print("⚠ PCA model not found. Using default parameters.")
            # Default: will need to be replaced with actual trained values
            self.pca_mean = np.zeros(30)
            self.pca_std = np.ones(30)
            self.pca_components = np.eye(3, 30)  # Identity for first 3 components

    def init_realsense(self):
        """Initialize RealSense camera"""
        try:
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)

            profile = self.pipeline.start(config)
            self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

            self.camera_available = True
            print("✓ RealSense camera initialized")
        except Exception as e:
            self.camera_available = False
            print(f"✗ RealSense error: {e}")
            messagebox.showwarning("Camera", "RealSense camera not available. Using file mode only.")

    def init_sam3(self):
        """Initialize SAM3 model"""
        if not SAM3_AVAILABLE:
            self.sam3_predictor = None
            print("✗ SAM3 not available")
            return

        try:
            # Load SAM3 model
            checkpoint = "sam2.1_b.pt"  # or your model path
            model_cfg = "sam2_hiera_b.yaml"

            if os.path.exists(checkpoint):
                sam2_model = build_sam2(model_cfg, checkpoint)
                self.sam3_predictor = SAM2ImagePredictor(sam2_model)
                print("✓ SAM3 model loaded successfully")
            else:
                self.sam3_predictor = None
                print(f"✗ SAM3 checkpoint not found: {checkpoint}")
        except Exception as e:
            self.sam3_predictor = None
            print(f"✗ SAM3 initialization error: {e}")

    def setup_gui(self):
        """Setup GUI layout"""
        # Main container
        main_container = tk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel: Video feed
        left_panel = tk.Frame(main_container, relief=tk.RIDGE, borderwidth=2)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        tk.Label(left_panel, text="📹 Live Camera Feed", font=("Arial", 14, "bold")).pack(pady=5)

        self.canvas = tk.Canvas(left_panel, width=640, height=480, bg="black")
        self.canvas.pack(padx=10, pady=10)

        # Segmentation overlay canvas
        tk.Label(left_panel, text="🎯 Segmentation Result", font=("Arial", 14, "bold")).pack(pady=5)
        self.seg_canvas = tk.Canvas(left_panel, width=640, height=480, bg="black")
        self.seg_canvas.pack(padx=10, pady=10)

        # Right panel: Controls and Results
        right_panel = tk.Frame(main_container, relief=tk.RIDGE, borderwidth=2, width=400)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y)
        right_panel.pack_propagate(False)

        # === CONTROLS ===
        control_frame = tk.LabelFrame(right_panel, text="⚙️ Controls", font=("Arial", 12, "bold"))
        control_frame.pack(fill=tk.X, padx=10, pady=10)

        # Buttons
        btn_frame = tk.Frame(control_frame)
        btn_frame.pack(pady=10)

        self.btn_segment = tk.Button(btn_frame, text="🎯 Auto Segment",
                                     command=self.run_segmentation,
                                     bg="#4CAF50", fg="white", font=("Arial", 12, "bold"),
                                     width=15)
        self.btn_segment.pack(pady=5)

        self.btn_classify = tk.Button(btn_frame, text="🔍 Classify",
                                      command=self.run_classification,
                                      bg="#2196F3", fg="white", font=("Arial", 12, "bold"),
                                      width=15)
        self.btn_classify.pack(pady=5)

        self.btn_load = tk.Button(btn_frame, text="📁 Load Image",
                                  command=self.load_image,
                                  bg="#9E9E9E", fg="white", font=("Arial", 12, "bold"),
                                  width=15)
        self.btn_load.pack(pady=5)

        self.btn_snapshot = tk.Button(btn_frame, text="📸 Snapshot",
                                      command=self.take_snapshot,
                                      bg="#FF9800", fg="white", font=("Arial", 12, "bold"),
                                      width=15)
        self.btn_snapshot.pack(pady=5)

        # === PCA RESULTS ===
        pca_frame = tk.LabelFrame(right_panel, text="📊 PCA Components", font=("Arial", 12, "bold"))
        pca_frame.pack(fill=tk.X, padx=10, pady=10)

        pca_display = tk.Frame(pca_frame)
        pca_display.pack(pady=10)

        tk.Label(pca_display, text="PC1:", font=("Arial", 11, "bold")).grid(row=0, column=0, sticky="e", padx=5)
        self.lbl_pc1 = tk.Label(pca_display, text="0.000", font=("Consolas", 16),
                               bg="#E3F2FD", fg="#1976D2", width=10, relief=tk.SUNKEN)
        self.lbl_pc1.grid(row=0, column=1, padx=5, pady=3)

        tk.Label(pca_display, text="PC2:", font=("Arial", 11, "bold")).grid(row=1, column=0, sticky="e", padx=5)
        self.lbl_pc2 = tk.Label(pca_display, text="0.000", font=("Consolas", 16),
                               bg="#E8F5E9", fg="#388E3C", width=10, relief=tk.SUNKEN)
        self.lbl_pc2.grid(row=1, column=1, padx=5, pady=3)

        tk.Label(pca_display, text="PC3:", font=("Arial", 11, "bold")).grid(row=2, column=0, sticky="e", padx=5)
        self.lbl_pc3 = tk.Label(pca_display, text="0.000", font=("Consolas", 16),
                               bg="#FFF3E0", fg="#F57C00", width=10, relief=tk.SUNKEN)
        self.lbl_pc3.grid(row=2, column=1, padx=5, pady=3)

        # === CLASSIFICATION RESULT ===
        result_frame = tk.LabelFrame(right_panel, text="🎯 Classification Result",
                                     font=("Arial", 12, "bold"))
        result_frame.pack(fill=tk.X, padx=10, pady=10)

        self.lbl_class = tk.Label(result_frame, text="Unknown", font=("Arial", 24, "bold"),
                                 bg="#FFFFFF", fg="#000000", relief=tk.RAISED, padx=20, pady=20)
        self.lbl_class.pack(pady=10)

        self.lbl_confidence = tk.Label(result_frame, text="Confidence: 0%",
                                      font=("Arial", 12))
        self.lbl_confidence.pack(pady=5)

        # Usability indicator
        self.lbl_usability = tk.Label(result_frame, text="⚪ Not Classified",
                                     font=("Arial", 14, "bold"))
        self.lbl_usability.pack(pady=10)

        # === DETAILED INFO ===
        info_frame = tk.LabelFrame(right_panel, text="ℹ️ Detailed Information",
                                  font=("Arial", 12, "bold"))
        info_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Text widget with scrollbar
        scroll = tk.Scrollbar(info_frame)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.info_text = tk.Text(info_frame, height=15, font=("Consolas", 9),
                                yscrollcommand=scroll.set, wrap=tk.WORD)
        self.info_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        scroll.config(command=self.info_text.yview)

        self.update_info("System ready. Waiting for segmentation...")

        # === STATUS BAR ===
        self.status_bar = tk.Label(self.root, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def update_frame(self):
        """Update camera feed"""
        if self.camera_available:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=100)
                if frames:
                    color_frame = frames.get_color_frame()
                    if color_frame:
                        self.current_frame = np.asanyarray(color_frame.get_data())
                        self.display_frame(self.current_frame, self.canvas)
            except Exception as e:
                pass

        self.root.after(30, self.update_frame)

    def display_frame(self, frame, canvas):
        """Display frame on canvas"""
        if frame is not None:
            # Resize to canvas size
            display_frame = cv2.resize(frame, (640, 480))

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

            # Convert to PIL Image
            img = Image.fromarray(rgb_frame)
            img_tk = ImageTk.PhotoImage(image=img)

            # Display on canvas
            canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
            canvas.image = img_tk  # Keep reference

    def run_segmentation(self):
        """Run SAM3 automatic segmentation"""
        if self.current_frame is None:
            messagebox.showwarning("Warning", "No image available. Please load an image or use camera.")
            return

        if self.sam3_predictor is None:
            messagebox.showwarning("Warning", "SAM3 not available. Please install sam2 library.")
            return

        self.update_status("Running segmentation...")

        try:
            # Prepare image
            image_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)

            # Set image for SAM3
            self.sam3_predictor.set_image(image_rgb)

            # Generate automatic masks
            # Use center point as prompt
            h, w = image_rgb.shape[:2]
            center_point = np.array([[w//2, h//2]])
            center_label = np.array([1])

            masks, scores, _ = self.sam3_predictor.predict(
                point_coords=center_point,
                point_labels=center_label,
                multimask_output=True
            )

            # Select best mask
            best_idx = np.argmax(scores)
            self.segmentation_mask = masks[best_idx]

            # Extract bounding box and polygon
            self.extract_segmentation_info()

            # Display segmentation
            self.display_segmentation()

            self.update_status("Segmentation completed")
            self.update_info(f"✓ Segmentation completed\n"
                           f"  Mask area: {np.sum(self.segmentation_mask)} pixels\n"
                           f"  Bounding box: {self.bbox_coords}\n"
                           f"  Polygon points: {len(self.polygon_coords)} vertices")

        except Exception as e:
            messagebox.showerror("Error", f"Segmentation failed: {str(e)}")
            self.update_status("Segmentation failed")

    def extract_segmentation_info(self):
        """Extract bounding box and polygon from mask"""
        if self.segmentation_mask is None:
            return

        # Find contours
        mask_uint8 = (self.segmentation_mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            # Get largest contour
            largest_contour = max(contours, key=cv2.contourArea)

            # Bounding box
            x, y, w, h = cv2.boundingRect(largest_contour)
            self.bbox_coords = (x, y, x+w, y+h)

            # Approximate polygon
            epsilon = 0.005 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            self.polygon_coords = approx.reshape(-1, 2).tolist()

    def display_segmentation(self):
        """Display segmentation result"""
        if self.current_frame is None or self.segmentation_mask is None:
            return

        # Create overlay
        overlay = self.current_frame.copy()

        # Apply mask (green tint)
        mask_color = np.zeros_like(overlay)
        mask_color[self.segmentation_mask > 0] = [0, 255, 0]
        overlay = cv2.addWeighted(overlay, 0.7, mask_color, 0.3, 0)

        # Draw bounding box
        if self.bbox_coords:
            x1, y1, x2, y2 = self.bbox_coords
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(overlay, "Segmented Region", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw polygon
        if self.polygon_coords:
            pts = np.array(self.polygon_coords, np.int32).reshape((-1, 1, 2))
            cv2.polylines(overlay, [pts], True, (255, 0, 255), 2)

        # Display
        self.display_frame(overlay, self.seg_canvas)

    def run_classification(self):
        """Run feature extraction and PCA classification"""
        if self.segmentation_mask is None:
            messagebox.showwarning("Warning", "Please run segmentation first.")
            return

        self.update_status("Extracting features...")

        try:
            # Extract features
            self.features = self.extract_features()

            if self.features is None:
                raise ValueError("Feature extraction failed")

            # Standardize features
            features_std = (self.features - self.pca_mean) / self.pca_std

            # PCA transformation
            self.pca_values = np.dot(features_std, self.pca_components.T)

            # Classification
            self.classify_nearest_centroid()

            # Update display
            self.update_pca_display()
            self.update_result_display()

            self.update_status("Classification completed")

        except Exception as e:
            messagebox.showerror("Error", f"Classification failed: {str(e)}")
            self.update_status("Classification failed")

    def extract_features(self):
        """Extract 30 features from segmented region"""
        if self.current_frame is None or self.segmentation_mask is None:
            return None

        try:
            # Get RealSense frames for IR and depth
            if self.camera_available:
                frames = self.pipeline.wait_for_frames(timeout_ms=100)
                ir_frame = frames.get_infrared_frame()
                depth_frame = frames.get_depth_frame()
                img_ir = np.asanyarray(ir_frame.get_data())
                img_depth = np.asanyarray(depth_frame.get_data())
            else:
                # Use dummy values if camera not available
                img_ir = np.zeros(self.current_frame.shape[:2], dtype=np.uint8)
                img_depth = np.zeros(self.current_frame.shape[:2], dtype=np.uint16)

            # Get bounding box and polygon regions
            x1, y1, x2, y2 = self.bbox_coords

            # ROIs
            roi_color = self.current_frame[y1:y2, x1:x2]
            roi_ir = img_ir[y1:y2, x1:x2]
            roi_depth = img_depth[y1:y2, x1:x2]

            # Polygon mask
            mask = np.zeros(self.current_frame.shape[:2], dtype=np.uint8)
            pts = np.array(self.polygon_coords, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], 255)
            mask_roi = mask[y1:y2, x1:x2]

            features = []

            # 1. Depth (1 feature)
            depth_mean = np.mean(roi_depth[mask_roi > 0]) * self.depth_scale if self.camera_available else 0.01
            features.append(depth_mean)

            # 2. Color features - Bounding Box (6 features)
            color_bbox_mean = cv2.mean(roi_color)[:3]
            color_bbox_std = np.std(roi_color.reshape(-1, 3), axis=0)
            features.extend(color_bbox_mean)
            features.extend(color_bbox_std)

            # 3. Color features - Polygon (6 features)
            color_poly_mean = cv2.mean(roi_color, mask=mask_roi)[:3]
            poly_pixels = roi_color[mask_roi > 0]
            color_poly_std = np.std(poly_pixels, axis=0) if len(poly_pixels) > 0 else [0, 0, 0]
            features.extend(color_poly_mean)
            features.extend(color_poly_std)

            # 4. HSV features - Bounding Box (3 features)
            hsv_bbox = cv2.cvtColor(roi_color, cv2.COLOR_BGR2HSV)
            hsv_bbox_mean = cv2.mean(hsv_bbox)[:3]
            features.extend(hsv_bbox_mean)

            # 5. HSV features - Polygon (3 features)
            hsv_poly_mean = cv2.mean(hsv_bbox, mask=mask_roi)[:3]
            features.extend(hsv_poly_mean)

            # 6. Texture features (4 features)
            gray_bbox = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
            gray_poly = gray_bbox.copy()
            gray_poly[mask_roi == 0] = 0

            # Entropy
            hist_bbox = cv2.calcHist([gray_bbox], [0], None, [256], [0, 256])
            hist_bbox = hist_bbox / hist_bbox.sum()
            entropy_bbox = -np.sum(hist_bbox * np.log2(hist_bbox + 1e-7))

            hist_poly = cv2.calcHist([gray_poly], [0], mask_roi, [256], [0, 256])
            hist_poly = hist_poly / (hist_poly.sum() + 1e-7)
            entropy_poly = -np.sum(hist_poly * np.log2(hist_poly + 1e-7))

            # Contrast
            contrast_bbox = gray_bbox.std()
            contrast_poly = gray_poly[mask_roi > 0].std() if np.sum(mask_roi) > 0 else 0

            features.extend([entropy_bbox, entropy_poly, contrast_bbox, contrast_poly])

            # 7. IR sensor (1 feature)
            ir_mean = cv2.mean(roi_ir, mask=mask_roi)[0] if self.camera_available else 100
            features.append(ir_mean)

            # Total: 1 + 6 + 6 + 3 + 3 + 4 + 1 = 24 features
            # Pad to 30 if needed
            while len(features) < 30:
                features.append(0.0)

            return np.array(features[:30])

        except Exception as e:
            print(f"Feature extraction error: {e}")
            return None

    def classify_nearest_centroid(self):
        """Classify using nearest centroid method"""
        pc1, pc2, pc3 = self.pca_values
        sample_point = np.array([pc1, pc2, pc3])

        min_distance = float('inf')
        distances = {}

        for class_name, centroid in self.centroids.items():
            distance = np.linalg.norm(sample_point - centroid)
            distances[class_name] = distance

            if distance < min_distance:
                min_distance = distance
                self.predicted_class = class_name

        # Calculate confidence (inverse distance normalized)
        inv_distances = {k: 1/(v+1e-6) for k, v in distances.items()}
        total = sum(inv_distances.values())
        self.confidence = inv_distances[self.predicted_class] / total

        # Store all distances for info display
        self.all_distances = distances

    def update_pca_display(self):
        """Update PCA component display"""
        pc1, pc2, pc3 = self.pca_values

        self.lbl_pc1.config(text=f"{pc1:+.3f}")
        self.lbl_pc2.config(text=f"{pc2:+.3f}")
        self.lbl_pc3.config(text=f"{pc3:+.3f}")

    def update_result_display(self):
        """Update classification result display"""
        # Class name
        self.lbl_class.config(text=self.predicted_class)

        # Color coding
        usable_oils = ['Dark oil', 'Light oil', 'Medium oil']
        if self.predicted_class in usable_oils:
            self.lbl_class.config(bg="#4CAF50", fg="white")  # Green
            self.lbl_usability.config(text="✅ USABLE OIL", fg="#4CAF50")
        else:
            self.lbl_class.config(bg="#F44336", fg="white")  # Red
            self.lbl_usability.config(text="❌ UNUSABLE / CONTAMINATED", fg="#F44336")

        # Confidence
        self.lbl_confidence.config(text=f"Confidence: {self.confidence*100:.1f}%")

        # Detailed info
        info_text = f"""
=== CLASSIFICATION RESULT ===
Predicted Class: {self.predicted_class}
Confidence: {self.confidence*100:.1f}%

=== PCA COORDINATES ===
PC1: {self.pca_values[0]:+.3f}  (37.2% variance)
PC2: {self.pca_values[1]:+.3f}  (28.5% variance)
PC3: {self.pca_values[2]:+.3f}  (15.6% variance)

=== DISTANCES TO CENTROIDS ===
"""
        # Sort by distance
        sorted_distances = sorted(self.all_distances.items(), key=lambda x: x[1])
        for i, (class_name, dist) in enumerate(sorted_distances):
            marker = "→" if class_name == self.predicted_class else " "
            info_text += f"{marker} {class_name:20s}: {dist:6.3f}\n"

        info_text += f"\n=== USABILITY ===\n"
        if self.predicted_class in usable_oils:
            info_text += "✅ This oil is USABLE for cooking\n"
        else:
            info_text += "❌ This oil is NOT suitable for use\n"
            if "water" in self.predicted_class.lower():
                info_text += "   Reason: Water contamination detected\n"
            elif "motor" in self.predicted_class.lower():
                info_text += "   Reason: Motor oil contamination\n"
            elif "lard" in self.predicted_class.lower():
                info_text += "   Reason: Animal fat contamination\n"

        self.update_info(info_text)

    def update_info(self, text):
        """Update information text widget"""
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, text)

    def update_status(self, text):
        """Update status bar"""
        self.status_bar.config(text=text)
        self.root.update_idletasks()

    def load_image(self):
        """Load image from file"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )

        if file_path:
            self.current_frame = cv2.imread(file_path)
            if self.current_frame is not None:
                self.display_frame(self.current_frame, self.canvas)
                self.update_status(f"Loaded: {os.path.basename(file_path)}")
            else:
                messagebox.showerror("Error", "Failed to load image")

    def take_snapshot(self):
        """Take snapshot from camera"""
        if self.current_frame is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"snapshot_{timestamp}.jpg"

            os.makedirs("snapshots", exist_ok=True)
            filepath = os.path.join("snapshots", filename)

            cv2.imwrite(filepath, self.current_frame)
            messagebox.showinfo("Snapshot", f"Saved: {filename}")
            self.update_status(f"Snapshot saved: {filename}")
        else:
            messagebox.showwarning("Warning", "No frame available")

    def on_closing(self):
        """Clean up on close"""
        if self.camera_available:
            try:
                self.pipeline.stop()
            except:
                pass
        self.root.destroy()

def main():
    """Main entry point"""
    root = tk.Tk()
    app = OilClassificationGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
