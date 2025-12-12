"""
Oil Quality Classification GUI using SAM3 Segmentation and PCA Classification
Features:
- Automatic segmentation using SAM3
- Real-time feature extraction (12 features after multicollinearity removal)
- PCA transformation to 3 components
- Nearest centroid classification
- Display PC1, PC2, PC3 values and oil type
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import pyrealsense2 as rs
import pickle
from datetime import datetime

# Try to import SAM3
try:
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM3_AVAILABLE = True
except ImportError:
    print("Warning: SAM3 not available. Install with: pip install git+https://github.com/facebookresearch/segment-anything-2.git")
    SAM3_AVAILABLE = False


class OilClassificationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Oil Quality Classification System")
        self.root.geometry("1200x900")

        # Initialize variables
        self.current_frame = None
        self.segmentation_mask = None
        self.bbox_coords = None
        self.polygon_coords = None
        self.features = None
        self.pca_values = None
        self.classification_result = None

        # Load PCA model
        self.load_pca_model()

        # Initialize RealSense camera
        self.camera_available = self.init_realsense()

        # Initialize SAM3
        self.sam3_available = self.init_sam3()

        # Create GUI
        self.create_widgets()

        # Start video stream if camera available
        if self.camera_available:
            self.update_video_stream()

    def load_pca_model(self):
        """Load PCA model parameters and centroids"""
        try:
            with open('pca_model_params.pkl', 'rb') as f:
                model_data = pickle.load(f)

            self.pca_mean = model_data['mean']
            self.pca_std = model_data['std']
            self.pca_components = model_data['components']
            self.centroids = model_data['centroids']
            self.feature_columns = model_data['feature_columns']

            # Define usable vs unusable classes
            self.usable_classes = {'Dark oil', 'Light oil', 'Medium oil'}

            print(f"✓ PCA model loaded: {len(self.centroids)} classes")
            print(f"  Feature columns ({len(self.feature_columns)}): {self.feature_columns}")

        except FileNotFoundError:
            messagebox.showerror("Error",
                "PCA model file not found! Please run save_pca_model.py first.")
            self.root.quit()

    def init_realsense(self):
        """Initialize Intel RealSense D435 camera"""
        try:
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            config.enable_stream(rs.stream.infrared, 640, 480, rs.format.y8, 30)

            self.pipeline.start(config)
            print("✓ RealSense camera initialized")
            return True
        except Exception as e:
            print(f"✗ Camera initialization failed: {e}")
            return False

    def init_sam3(self):
        """Initialize SAM3 model"""
        if not SAM3_AVAILABLE:
            return False

        try:
            model_path = "sam2.1_b.pt"
            self.sam3_predictor = SAM2ImagePredictor.from_pretrained(
                "facebook/sam2.1-hiera-base-plus"
            )
            print("✓ SAM3 model loaded")
            return True
        except Exception as e:
            print(f"✗ SAM3 initialization failed: {e}")
            return False

    def create_widgets(self):
        """Create GUI widgets"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Left panel - Video feed
        left_frame = ttk.LabelFrame(main_frame, text="Camera Feed", padding="10")
        left_frame.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.video_label = ttk.Label(left_frame)
        self.video_label.pack()

        # Control buttons
        button_frame = ttk.Frame(left_frame)
        button_frame.pack(pady=10)

        ttk.Button(button_frame, text="Auto Segment",
                  command=self.run_segmentation).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Classify",
                  command=self.run_classification).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Load Image",
                  command=self.load_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Take Snapshot",
                  command=self.take_snapshot).pack(side=tk.LEFT, padx=5)

        # Right panel - Results
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=0, column=1, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))

        # PCA values
        pca_frame = ttk.LabelFrame(right_frame, text="PCA Components", padding="10")
        pca_frame.pack(fill=tk.X, pady=5)

        self.pc1_label = ttk.Label(pca_frame, text="PC1: --",
                                   font=("Arial", 12, "bold"),
                                   background="#E3F2FD", padding=5)
        self.pc1_label.pack(fill=tk.X, pady=2)

        self.pc2_label = ttk.Label(pca_frame, text="PC2: --",
                                   font=("Arial", 12, "bold"),
                                   background="#FFF3E0", padding=5)
        self.pc2_label.pack(fill=tk.X, pady=2)

        self.pc3_label = ttk.Label(pca_frame, text="PC3: --",
                                   font=("Arial", 12, "bold"),
                                   background="#F3E5F5", padding=5)
        self.pc3_label.pack(fill=tk.X, pady=2)

        # Classification result
        result_frame = ttk.LabelFrame(right_frame, text="Classification Result", padding="10")
        result_frame.pack(fill=tk.X, pady=5)

        self.result_label = ttk.Label(result_frame, text="Oil Type: --",
                                      font=("Arial", 14, "bold"), padding=10)
        self.result_label.pack(fill=tk.X)

        self.usability_label = ttk.Label(result_frame, text="",
                                         font=("Arial", 12), padding=5)
        self.usability_label.pack(fill=tk.X)

        self.confidence_label = ttk.Label(result_frame, text="Confidence: --",
                                          font=("Arial", 10), padding=5)
        self.confidence_label.pack(fill=tk.X)

        # Detailed information
        info_frame = ttk.LabelFrame(right_frame, text="Detailed Information", padding="10")
        info_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.info_text = tk.Text(info_frame, height=15, width=40, font=("Courier", 9))
        self.info_text.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(info_frame, command=self.info_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.info_text.config(yscrollcommand=scrollbar.set)

        # Status bar
        self.status_label = ttk.Label(main_frame, text="Ready",
                                      relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=2)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

    def update_video_stream(self):
        """Update video stream from camera"""
        if not self.camera_available:
            return

        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=100)
            color_frame = frames.get_color_frame()

            if color_frame:
                # Convert to numpy array
                frame = np.asanyarray(color_frame.get_data())
                self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Draw segmentation if available
                display_frame = self.current_frame.copy()
                if self.segmentation_mask is not None:
                    # Draw mask overlay
                    overlay = np.zeros_like(display_frame)
                    overlay[self.segmentation_mask] = [0, 255, 0]
                    display_frame = cv2.addWeighted(display_frame, 0.7, overlay, 0.3, 0)

                    # Draw bounding box
                    if self.bbox_coords is not None:
                        x1, y1, x2, y2 = self.bbox_coords
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                    # Draw polygon
                    if self.polygon_coords is not None:
                        pts = np.array(self.polygon_coords, np.int32).reshape((-1, 1, 2))
                        cv2.polylines(display_frame, [pts], True, (0, 255, 255), 2)

                # Display
                self.display_image(display_frame)

        except Exception as e:
            print(f"Video stream error: {e}")

        # Schedule next update
        self.root.after(30, self.update_video_stream)

    def display_image(self, image):
        """Display image on GUI"""
        image_pil = Image.fromarray(image)
        image_pil.thumbnail((640, 480))
        photo = ImageTk.PhotoImage(image_pil)
        self.video_label.config(image=photo)
        self.video_label.image = photo

    def run_segmentation(self):
        """Run SAM3 automatic segmentation"""
        if not self.sam3_available:
            messagebox.showwarning("Warning", "SAM3 not available")
            return

        if self.current_frame is None:
            messagebox.showwarning("Warning", "No image available")
            return

        self.update_status("Running segmentation...")

        try:
            # Use center point as prompt
            h, w = self.current_frame.shape[:2]
            center_point = np.array([[w//2, h//2]])
            center_label = np.array([1])

            # Convert RGB to BGR for SAM3
            image_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_RGB2BGR)

            # Run SAM3
            self.sam3_predictor.set_image(image_rgb)
            masks, scores, _ = self.sam3_predictor.predict(
                point_coords=center_point,
                point_labels=center_label,
                multimask_output=True
            )

            # Select best mask
            best_idx = np.argmax(scores)
            self.segmentation_mask = masks[best_idx]

            # Calculate bounding box
            y_indices, x_indices = np.where(self.segmentation_mask)
            x1, x2 = x_indices.min(), x_indices.max()
            y1, y2 = y_indices.min(), y_indices.max()
            self.bbox_coords = [x1, y1, x2, y2]

            # Calculate polygon (contour)
            mask_uint8 = (self.segmentation_mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                epsilon = 0.01 * cv2.arcLength(largest_contour, True)
                self.polygon_coords = cv2.approxPolyDP(largest_contour, epsilon, True)
                self.polygon_coords = self.polygon_coords.reshape(-1, 2).tolist()

            self.update_status(f"Segmentation complete (score: {scores[best_idx]:.3f})")

        except Exception as e:
            messagebox.showerror("Error", f"Segmentation failed: {e}")
            self.update_status("Segmentation failed")

    def run_classification(self):
        """Run feature extraction, PCA, and classification"""
        if self.segmentation_mask is None:
            messagebox.showwarning("Warning", "Please run segmentation first")
            return

        self.update_status("Extracting features...")

        # Extract features
        self.features = self.extract_features()
        if self.features is None:
            messagebox.showerror("Error", "Feature extraction failed")
            return

        self.update_status("Running PCA transformation...")

        # PCA transformation
        features_std = (self.features - self.pca_mean) / self.pca_std
        self.pca_values = np.dot(features_std, self.pca_components.T)

        # Update PCA display
        self.update_pca_display()

        self.update_status("Classifying...")

        # Classify
        self.classification_result = self.classify_nearest_centroid()

        # Update result display
        self.update_result_display()

        self.update_status("Classification complete")

    def extract_features(self):
        """Extract 12 features matching cleaned dataset"""
        if self.current_frame is None or self.segmentation_mask is None:
            return None

        try:
            # Get RealSense IR frame
            if self.camera_available:
                frames = self.pipeline.wait_for_frames(timeout_ms=100)
                ir_frame = frames.get_infrared_frame()
                img_ir = np.asanyarray(ir_frame.get_data())
            else:
                img_ir = np.ones(self.current_frame.shape[:2], dtype=np.uint8) * 150

            # Get bounding box region
            x1, y1, x2, y2 = self.bbox_coords

            # ROIs
            roi_color = self.current_frame[y1:y2, x1:x2]
            roi_ir = img_ir[y1:y2, x1:x2]

            # Polygon mask
            mask = np.zeros(self.current_frame.shape[:2], dtype=np.uint8)
            pts = np.array(self.polygon_coords, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], 255)
            mask_roi = mask[y1:y2, x1:x2]

            # Convert RGB to BGR for OpenCV
            roi_color_bgr = cv2.cvtColor(roi_color, cv2.COLOR_RGB2BGR)

            features = []

            # 1. color_bbox_mean_r
            color_bbox_mean_bgr = cv2.mean(roi_color_bgr)[:3]
            features.append(color_bbox_mean_bgr[2])  # R

            # 2. color_bbox_mean_g
            features.append(color_bbox_mean_bgr[1])  # G

            # 3. color_poly_mean_b
            color_poly_mean_bgr = cv2.mean(roi_color_bgr, mask=mask_roi)[:3]
            features.append(color_poly_mean_bgr[0])  # B

            # 4. color_poly_std_b
            poly_pixels_bgr = roi_color_bgr[mask_roi > 0]
            if len(poly_pixels_bgr) > 0:
                color_poly_std_bgr = np.std(poly_pixels_bgr, axis=0)
                features.append(color_poly_std_bgr[0])  # B std
            else:
                features.append(0.0)

            # 5-7. hsv_bbox_mean_h, hsv_bbox_mean_s, hsv_bbox_mean_v
            hsv_bbox = cv2.cvtColor(roi_color_bgr, cv2.COLOR_BGR2HSV)
            hsv_bbox_mean = cv2.mean(hsv_bbox)[:3]
            features.extend(hsv_bbox_mean)

            # 8-10. hsv_poly_mean_h, hsv_poly_mean_s, hsv_poly_mean_v
            hsv_poly_mean = cv2.mean(hsv_bbox, mask=mask_roi)[:3]
            features.extend(hsv_poly_mean)

            # 11. color_bbox_entropy
            gray_bbox = cv2.cvtColor(roi_color_bgr, cv2.COLOR_BGR2GRAY)
            hist_bbox = cv2.calcHist([gray_bbox], [0], None, [256], [0, 256])
            hist_bbox = hist_bbox / (hist_bbox.sum() + 1e-7)
            entropy_bbox = -np.sum(hist_bbox * np.log2(hist_bbox + 1e-7))
            features.append(entropy_bbox)

            # 12. color_poly_entropy
            hist_poly = cv2.calcHist([gray_bbox], [0], mask_roi, [256], [0, 256])
            hist_poly = hist_poly / (hist_poly.sum() + 1e-7)
            entropy_poly = -np.sum(hist_poly * np.log2(hist_poly + 1e-7))
            features.append(entropy_poly)

            # 13. ir_cluster (mean IR value)
            ir_mean = cv2.mean(roi_ir, mask=mask_roi)[0]
            features.append(ir_mean)

            return np.array(features)

        except Exception as e:
            print(f"Feature extraction error: {e}")
            return None

    def classify_nearest_centroid(self):
        """Classify using nearest centroid method"""
        pc1, pc2, pc3 = self.pca_values
        sample_point = np.array([pc1, pc2, pc3])

        distances = {}
        for class_name, centroid in self.centroids.items():
            distance = np.linalg.norm(sample_point - centroid)
            distances[class_name] = distance

        # Find nearest
        predicted_class = min(distances, key=distances.get)
        min_distance = distances[predicted_class]

        return {
            'class': predicted_class,
            'distance': min_distance,
            'distances': distances,
            'usable': predicted_class in self.usable_classes
        }

    def update_pca_display(self):
        """Update PCA values display"""
        pc1, pc2, pc3 = self.pca_values
        self.pc1_label.config(text=f"PC1: {pc1:+.3f}")
        self.pc2_label.config(text=f"PC2: {pc2:+.3f}")
        self.pc3_label.config(text=f"PC3: {pc3:+.3f}")

    def update_result_display(self):
        """Update classification result display"""
        result = self.classification_result

        # Oil type
        self.result_label.config(text=f"Oil Type: {result['class']}")

        # Usability
        if result['usable']:
            usability_text = "USABLE OIL"
            bg_color = "#C8E6C9"  # Light green
        else:
            usability_text = "UNUSABLE OIL"
            bg_color = "#FFCDD2"  # Light red

        self.usability_label.config(text=usability_text, background=bg_color)

        # Confidence (inverse of distance)
        confidence = 100 * np.exp(-result['distance'])
        self.confidence_label.config(text=f"Confidence: {confidence:.1f}%")

        # Detailed info
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, "=== Classification Details ===\n\n")
        self.info_text.insert(tk.END, f"Predicted Class: {result['class']}\n")
        self.info_text.insert(tk.END, f"Distance to centroid: {result['distance']:.3f}\n\n")

        self.info_text.insert(tk.END, "=== PCA Values ===\n")
        self.info_text.insert(tk.END, f"PC1: {self.pca_values[0]:+.3f}\n")
        self.info_text.insert(tk.END, f"PC2: {self.pca_values[1]:+.3f}\n")
        self.info_text.insert(tk.END, f"PC3: {self.pca_values[2]:+.3f}\n\n")

        self.info_text.insert(tk.END, "=== Distances to All Centroids ===\n")
        sorted_distances = sorted(result['distances'].items(), key=lambda x: x[1])
        for cls, dist in sorted_distances:
            marker = " <--" if cls == result['class'] else ""
            self.info_text.insert(tk.END, f"{cls:15s}: {dist:6.3f}{marker}\n")

        self.info_text.insert(tk.END, f"\n=== Features ({len(self.features)}) ===\n")
        for i, (name, val) in enumerate(zip(self.feature_columns, self.features)):
            self.info_text.insert(tk.END, f"{name:20s}: {val:8.3f}\n")

    def load_image(self):
        """Load image from file"""
        filename = filedialog.askopenfilename(
            title="Select image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*")]
        )

        if filename:
            image = cv2.imread(filename)
            self.current_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.display_image(self.current_frame)
            self.segmentation_mask = None
            self.update_status(f"Loaded: {filename}")

    def take_snapshot(self):
        """Save current frame"""
        if self.current_frame is None:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"snapshot_{timestamp}.jpg"

        cv2.imwrite(filename, cv2.cvtColor(self.current_frame, cv2.COLOR_RGB2BGR))
        self.update_status(f"Saved: {filename}")

    def update_status(self, message):
        """Update status bar"""
        self.status_label.config(text=message)
        self.root.update_idletasks()

    def cleanup(self):
        """Cleanup resources"""
        if self.camera_available:
            self.pipeline.stop()


def main():
    root = tk.Tk()
    app = OilClassificationGUI(root)

    def on_closing():
        app.cleanup()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
