"""
Oil Quality Classification GUI using SAM3 Segmentation and PCA Classification
Features:
- Automatic segmentation using SAM3
- Real-time feature extraction (12 features)
- PCA transformation & Classification
- [NEW] Yellow Score Calculation (Min(R,G) - B)
- [NEW] Black Score Calculation (Inverted Luminance)
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import pyrealsense2 as rs
import pickle
from datetime import datetime
from auto_segmentation import AutoSegmenter

# --- CONFIGURATION ---
ROTATE_PREVIEW = True
ROTATE_DIR = cv2.ROTATE_90_COUNTERCLOCKWISE


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
        
        # Custom Scores
        self.current_yellow_score = 0.0
        self.current_black_score = 0.0  # [NEW]

        # Auto-segmentation variables
        self.auto_seg_results = None
        self.current_clusters = {}

        # Manual BBox variables
        self.manual_bbox_mode = False
        self.drawing = False
        self.start_x = 0
        self.start_y = 0
        self.temp_bbox = None
        self.display_scale = 1.0

        # IR cluster thresholds for each oil type
        self.ir_thresholds = {
            'Water': 182.40,
            'Mix (Oil+Water)': 181.01,
            'Mix (Oil+Motor)': 47.12,
            'Mix (Oil+Lard)': 91.52,
            'Lard': 79.50,
            'Motor oil': 47.16,
            'Medium oil': 174.76,
            'Light oil': 178.98,
            'Dark oil': 141.84
        }

        # Load PCA model
        self.load_pca_model()

        # Initialize RealSense camera
        self.camera_available = self.init_realsense()

        # Initialize Auto Segmenter
        self.auto_segmenter = AutoSegmenter(use_sam=True)
        self.sam3_available = self.auto_segmenter.sam_available

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

            self.usable_classes = {'Dark oil', 'Light oil', 'Medium oil'}

            print(f"✓ PCA model loaded: {len(self.centroids)} classes")

        except FileNotFoundError:
            messagebox.showerror("Error", "PCA model file not found!")
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

    def create_widgets(self):
        """Create GUI widgets"""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Left panel - Video feed
        left_frame = ttk.LabelFrame(main_frame, text="Camera Feed", padding="10")
        left_frame.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.video_label = ttk.Label(left_frame)
        self.video_label.pack()

        # Segmentation Prompt Selection
        prompt_frame = ttk.Frame(left_frame)
        prompt_frame.pack(pady=5)
        ttk.Label(prompt_frame, text="Seg Prompt:").pack(side=tk.LEFT, padx=5)
        self.segmentation_prompt = tk.StringVar(value="bottle")
        ttk.Radiobutton(prompt_frame, text="Bottle (Split)", variable=self.segmentation_prompt, value="bottle").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(prompt_frame, text="Liquid (Whole)", variable=self.segmentation_prompt, value="Liquid").pack(side=tk.LEFT, padx=5)

        # Control buttons
        button_frame = ttk.Frame(left_frame)
        button_frame.pack(pady=10)

        ttk.Button(button_frame, text="Auto Segment", command=self.run_segmentation).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cluster Region", command=self.run_clustering_on_mask).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Middle 1/3 Mask", command=self.refine_mask_middle_third).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Manual BBox", command=self.toggle_manual_bbox).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Yellow Liquid Detect", command=self.detect_yellow_liquid, style="Warning.TButton").pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Take Snapshot", command=self.take_snapshot).pack(side=tk.LEFT, padx=5)

        # Bind mouse events for manual bbox
        self.video_label.bind("<ButtonPress-1>", self.on_mouse_down)
        self.video_label.bind("<B1-Motion>", self.on_mouse_move)
        self.video_label.bind("<ButtonRelease-1>", self.on_mouse_up)

        # Cluster visibility
        cluster_vis_frame = ttk.LabelFrame(left_frame, text="Cluster Display", padding="10")
        cluster_vis_frame.pack(pady=5, fill=tk.X)

        self.show_cluster_0 = tk.BooleanVar(value=True)
        self.show_cluster_1 = tk.BooleanVar(value=True)

        ttk.Checkbutton(cluster_vis_frame, text="Show Cluster 0 (Green)", variable=self.show_cluster_0).pack(anchor=tk.W)
        ttk.Checkbutton(cluster_vis_frame, text="Show Cluster 1 (Magenta)", variable=self.show_cluster_1).pack(anchor=tk.W)

        # Cluster selection
        cluster_select_frame = ttk.LabelFrame(left_frame, text="Select Cluster for Classification", padding="10")
        cluster_select_frame.pack(pady=5, fill=tk.X)

        self.selected_cluster_for_classification = tk.StringVar(value="auto")

        ttk.Radiobutton(cluster_select_frame, text="Auto (Larger Area)", variable=self.selected_cluster_for_classification, value="auto").pack(anchor=tk.W)
        ttk.Radiobutton(cluster_select_frame, text="Cluster 0 (Green)", variable=self.selected_cluster_for_classification, value="cluster_0").pack(anchor=tk.W)
        ttk.Radiobutton(cluster_select_frame, text="Cluster 1 (Magenta)", variable=self.selected_cluster_for_classification, value="cluster_1").pack(anchor=tk.W)

        self.cluster_0_info = ttk.Label(cluster_select_frame, text="Cluster 0: N/A", font=("Consolas", 9), foreground="green")
        self.cluster_0_info.pack(anchor=tk.W, pady=(5,0))
        self.cluster_1_info = ttk.Label(cluster_select_frame, text="Cluster 1: N/A", font=("Consolas", 9), foreground="purple")
        self.cluster_1_info.pack(anchor=tk.W)

        # Method selection
        method_frame = ttk.LabelFrame(left_frame, text="Classification Method", padding="10")
        method_frame.pack(pady=5, fill=tk.X)

        self.classification_method = tk.StringVar(value="ir")
        ttk.Radiobutton(method_frame, text="IR Cluster (Recommended)", variable=self.classification_method, value="ir").pack(anchor=tk.W)
        ttk.Radiobutton(method_frame, text="PCA Model", variable=self.classification_method, value="pca").pack(anchor=tk.W)

        ttk.Button(cluster_select_frame, text="🔍 Classify Selected Cluster", command=self.run_classification).pack(pady=10, fill=tk.X)

        # Right panel - Results
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=0, column=1, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))

        # PCA values
        pca_frame = ttk.LabelFrame(right_frame, text="PCA Components", padding="10")
        pca_frame.pack(fill=tk.X, pady=5)

        self.pc1_label = ttk.Label(pca_frame, text="PC1: --", font=("Arial", 12, "bold"), background="#E3F2FD", padding=5)
        self.pc1_label.pack(fill=tk.X, pady=2)
        self.pc2_label = ttk.Label(pca_frame, text="PC2: --", font=("Arial", 12, "bold"), background="#FFF3E0", padding=5)
        self.pc2_label.pack(fill=tk.X, pady=2)
        self.pc3_label = ttk.Label(pca_frame, text="PC3: --", font=("Arial", 12, "bold"), background="#F3E5F5", padding=5)
        self.pc3_label.pack(fill=tk.X, pady=2)

        # Result
        result_frame = ttk.LabelFrame(right_frame, text="Classification Result", padding="10")
        result_frame.pack(fill=tk.X, pady=5)

        self.result_label = ttk.Label(result_frame, text="Oil Type: --", font=("Arial", 14, "bold"), padding=10)
        self.result_label.pack(fill=tk.X)
        self.usability_label = ttk.Label(result_frame, text="", font=("Arial", 12), padding=5)
        self.usability_label.pack(fill=tk.X)
        self.confidence_label = ttk.Label(result_frame, text="Confidence: --", font=("Arial", 10), padding=5)
        self.confidence_label.pack(fill=tk.X)

        # Info
        info_frame = ttk.LabelFrame(right_frame, text="Detailed Information", padding="10")
        info_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.info_text = tk.Text(info_frame, height=15, width=40, font=("Courier", 9))
        self.info_text.pack(fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(info_frame, command=self.info_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.info_text.config(yscrollcommand=scrollbar.set)

        self.status_label = ttk.Label(main_frame, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E))

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=2)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

    def update_video_stream(self):
        if not self.camera_available: return
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=100)
            color_frame = frames.get_color_frame()
            if color_frame:
                frame = np.asanyarray(color_frame.get_data())
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if ROTATE_PREVIEW: frame_rgb = cv2.rotate(frame_rgb, ROTATE_DIR)
                self.current_frame = frame_rgb
                
                display_frame = self.current_frame.copy()
                if self.current_clusters:
                    if self.show_cluster_0.get():
                        cluster_0 = self.current_clusters.get('cluster_0', {})
                        if cluster_0.get('mask') is not None:
                            contours, _ = cv2.findContours(cluster_0['mask'].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            cv2.drawContours(display_frame, contours, -1, (0, 255, 0), 2)
                            if cluster_0['bbox'] != (0, 0, 0, 0):
                                x1, y1, x2, y2 = cluster_0['bbox']
                                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    if self.show_cluster_1.get():
                        cluster_1 = self.current_clusters.get('cluster_1', {})
                        if cluster_1.get('mask') is not None:
                            contours, _ = cv2.findContours(cluster_1['mask'].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            cv2.drawContours(display_frame, contours, -1, (255, 0, 255), 2)
                elif self.segmentation_mask is not None:
                    overlay = np.zeros_like(display_frame)
                    overlay[self.segmentation_mask] = [0, 255, 0]
                    display_frame = cv2.addWeighted(display_frame, 0.7, overlay, 0.3, 0)
                    if self.bbox_coords:
                        x1, y1, x2, y2 = self.bbox_coords
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                # Draw manual bbox being drawn
                if self.manual_bbox_mode and self.temp_bbox:
                    tx1, ty1, tx2, ty2 = self.temp_bbox
                    # Convert screen coords to frame coords for drawing
                    s = self.display_scale
                    cv2.rectangle(display_frame, 
                                (int(tx1/s), int(ty1/s)), 
                                (int(tx2/s), int(ty2/s)), 
                                (0, 0, 255), 2)

                self.display_image(display_frame)
        except Exception as e: print(f"Video stream error: {e}")
        self.root.after(30, self.update_video_stream)

    def display_image(self, image):
        image_pil = Image.fromarray(image)
        orig_w, orig_h = image_pil.size
        
        if ROTATE_PREVIEW: target_size = (360, 480)
        else: target_size = (640, 480)
        
        image_pil.thumbnail(target_size)
        new_w, new_h = image_pil.size
        self.display_scale = new_w / orig_w if orig_w > 0 else 1.0
        
        photo = ImageTk.PhotoImage(image_pil)
        self.video_label.config(image=photo)
        self.video_label.image = photo

    def toggle_manual_bbox(self):
        self.manual_bbox_mode = not self.manual_bbox_mode
        if self.manual_bbox_mode:
            self.update_status("Manual BBox Mode: ON - Click and drag to select area")
            self.video_label.config(cursor="cross")
        else:
            self.update_status("Manual BBox Mode: OFF")
            self.video_label.config(cursor="")
            self.temp_bbox = None

    def on_mouse_down(self, event):
        if not self.manual_bbox_mode or self.current_frame is None: return
        self.drawing = True
        self.start_x = event.x
        self.start_y = event.y
        self.temp_bbox = (event.x, event.y, event.x, event.y)

    def on_mouse_move(self, event):
        if not self.drawing: return
        self.temp_bbox = (self.start_x, self.start_y, event.x, event.y)

    def on_mouse_up(self, event):
        if not self.drawing: return
        self.drawing = False
        
        x1, y1 = self.start_x, self.start_y
        x2, y2 = event.x, event.y
        
        # Normalize coordinates
        x_min, x_max = sorted([x1, x2])
        y_min, y_max = sorted([y1, y2])
        
        # Convert to frame coordinates
        scale = self.display_scale
        fx_min = int(x_min / scale)
        fx_max = int(x_max / scale)
        fy_min = int(y_min / scale)
        fy_max = int(y_max / scale)
        
        # Clamp to frame dimensions
        h, w = self.current_frame.shape[:2]
        fx_min = max(0, min(fx_min, w))
        fx_max = max(0, min(fx_max, w))
        fy_min = max(0, min(fy_min, h))
        fy_max = max(0, min(fy_max, h))
        
        if (fx_max - fx_min) > 10 and (fy_max - fy_min) > 10:
            self.bbox_coords = (fx_min, fy_min, fx_max, fy_max)
            
            # Create a simple rectangular mask
            self.segmentation_mask = np.zeros((h, w), dtype=np.uint8)
            self.segmentation_mask[fy_min:fy_max, fx_min:fx_max] = 1
            
            # Clear polygon coords as we don't have a polygon for manual box
            self.polygon_coords = None
            
            # Clear auto seg results to avoid confusion
            self.auto_seg_results = None
            self.current_clusters = {}
            
            self.update_status(f"Manual BBox set: {self.bbox_coords}")
            self.manual_bbox_mode = False
            self.video_label.config(cursor="")
        else:
            self.update_status("Selection too small")
            
        self.temp_bbox = None

    def run_segmentation(self):
        if self.current_frame is None: return
        
        prompt = self.segmentation_prompt.get()
        n_clusters = 1 if prompt == "Liquid" else 2
        
        self.update_status(f"Running auto-segmentation ({prompt})...")
        try:
            image_bgr = cv2.cvtColor(self.current_frame, cv2.COLOR_RGB2BGR)
            self.auto_seg_results = self.auto_segmenter.process_frame(image_bgr, prompt=prompt, n_clusters=n_clusters)
            self.update_cluster_data()
            
            if self.auto_seg_results and self.auto_seg_results['segmentation_mask'] is not None:
                # Select larger cluster
                c0 = self.current_clusters.get('cluster_0', {})
                c1 = self.current_clusters.get('cluster_1', {})
                a0 = np.sum(c0.get('mask', 0)) if c0.get('mask') is not None else 0
                a1 = np.sum(c1.get('mask', 0)) if c1.get('mask') is not None else 0
                
                if n_clusters == 1:
                    # If only 1 cluster requested, use cluster_0
                    sel, sel_name = c0, 'cluster_0'
                else:
                    if a0 >= a1 and a0 > 0: sel, sel_name = c0, 'cluster_0'
                    elif a1 > 0: sel, sel_name = c1, 'cluster_1'
                    else: sel = None

                if sel and sel.get('mask') is not None:
                    self.segmentation_mask = sel['mask']
                    self.bbox_coords = sel['bbox']
                    self.polygon_coords = self.auto_seg_results.get('polygons', {}).get(sel_name, [])
                    self.update_status(f"Segmentation complete (using {sel_name})")
                else: messagebox.showwarning("Warning", "No valid segmentation")
            else: messagebox.showwarning("Warning", "Segmentation failed")
        except Exception as e: messagebox.showerror("Error", f"Seg failed: {e}")

    def run_clustering_on_mask(self):
        if self.current_frame is None or self.segmentation_mask is None: return
        self.update_status("Running clustering...")
        try:
            image_bgr = cv2.cvtColor(self.current_frame, cv2.COLOR_RGB2BGR)
            clustering = self.auto_segmenter.apply_kmeans_clustering(image_bgr, self.segmentation_mask, n_clusters=2)
            bboxes = self.auto_segmenter.get_cluster_bboxes(clustering['cluster_masks'])
            polygons = self.auto_segmenter.get_cluster_polygons(clustering['cluster_masks'])
            
            self.auto_seg_results = {
                'segmentation_mask': self.segmentation_mask,
                'clustering': clustering,
                'bboxes': bboxes,
                'polygons': polygons
            }
            self.update_cluster_data()
            self.update_status("Clustering complete")
        except Exception as e: messagebox.showerror("Error", f"Clustering failed: {e}")

    def refine_mask_middle_third(self):
        """Keep only the middle 1/3 (vertically) of the current mask"""
        if self.segmentation_mask is None:
            messagebox.showwarning("Warning", "No mask available")
            return

        try:
            # Get bounding box of current mask
            rows = np.any(self.segmentation_mask, axis=1)
            cols = np.any(self.segmentation_mask, axis=0)
            
            if not np.any(rows) or not np.any(cols):
                messagebox.showwarning("Warning", "Mask is empty")
                return
                
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]
            
            height = y_max - y_min
            
            # Calculate middle 1/3 vertical range
            new_y_min = int(y_min + height / 3)
            new_y_max = int(y_min + 2 * height / 3)
            
            # Create new mask
            new_mask = np.zeros_like(self.segmentation_mask)
            new_mask[new_y_min:new_y_max, :] = self.segmentation_mask[new_y_min:new_y_max, :]
            
            # Update state
            self.segmentation_mask = new_mask
            
            # Update bbox
            rows_new = np.any(new_mask, axis=1)
            cols_new = np.any(new_mask, axis=0)
            if np.any(rows_new) and np.any(cols_new):
                ny_min, ny_max = np.where(rows_new)[0][[0, -1]]
                nx_min, nx_max = np.where(cols_new)[0][[0, -1]]
                self.bbox_coords = (nx_min, ny_min, nx_max, ny_max)
            
            # Clear polygons as they are no longer valid for the sliced mask
            self.polygon_coords = None
            
            # Clear auto seg results to avoid confusion
            self.auto_seg_results = None
            self.current_clusters = {}
            
            self.update_status(f"Mask refined to middle 1/3 (H: {height} -> {new_y_max - new_y_min})")
            
        except Exception as e:
            messagebox.showerror("Error", f"Refinement failed: {e}")

    def detect_yellow_liquid(self):
        if self.current_frame is None: return
        self.update_status("Detecting yellow liquid...")
        try:
            image_bgr = cv2.cvtColor(self.current_frame, cv2.COLOR_RGB2BGR)
            res = self.auto_segmenter.process_frame(image_bgr, prompt="yellow liquid", n_clusters=2)
            if res and res['segmentation_mask'] is not None and np.sum(res['segmentation_mask']) > 100:
                self.auto_seg_results = res
                self.update_cluster_data()
                self.segmentation_mask = res['segmentation_mask']
                self.update_status(f"✓ Yellow liquid detected!")
                messagebox.showinfo("Success", f"Yellow liquid detected!")
            else: messagebox.showwarning("Failed", "Yellow liquid area too small or not found.")
        except Exception as e: messagebox.showerror("Error", f"Detection failed: {e}")

    def update_cluster_data(self):
        if not self.auto_seg_results: return
        cl = self.auto_seg_results['clustering']
        bb = self.auto_seg_results['bboxes']
        
        self.current_clusters = {
            'cluster_0': {'bbox': bb.get('cluster_0', (0,0,0,0)), 'mask': cl['cluster_masks'].get('cluster_0'), 'stats': cl['cluster_stats'].get('cluster_0', {})},
            'cluster_1': {'bbox': bb.get('cluster_1', (0,0,0,0)), 'mask': cl['cluster_masks'].get('cluster_1'), 'stats': cl['cluster_stats'].get('cluster_1', {})}
        }
        
        c0, c1 = self.current_clusters['cluster_0'], self.current_clusters['cluster_1']
        a0 = np.sum(c0.get('mask', 0)) if c0.get('mask') is not None else 0
        a1 = np.sum(c1.get('mask', 0)) if c1.get('mask') is not None else 0
        self.cluster_0_info.config(text=f"Cluster 0: {a0} px ({c0.get('stats',{}).get('percentage',0):.1f}%)")
        self.cluster_1_info.config(text=f"Cluster 1: {a1} px ({c1.get('stats',{}).get('percentage',0):.1f}%)")

    def run_classification(self):
        sel = self.selected_cluster_for_classification.get()
        if sel == "auto" and self.segmentation_mask is None:
            messagebox.showwarning("Warning", "Run segmentation first")
            return
            
        if sel != "auto":
            c = self.current_clusters.get(sel, {})
            if c.get('mask') is None or np.sum(c.get('mask', 0)) == 0:
                messagebox.showwarning("Warning", f"{sel} not available")
                return
            self.segmentation_mask = c['mask']
            self.bbox_coords = c['bbox']

        self.features = self.extract_features()
        if self.features is None: return

        method = self.classification_method.get()
        if method == "ir":
            self.update_status("Classifying (IR)...")
            ir_val = self.features[11]
            self.classification_result = self.classify_by_ir_cluster(ir_val)
            self.pca_values = np.array([0.0, 0.0, ir_val])
            self.pc1_label.config(text="PC1: N/A")
            self.pc2_label.config(text="PC2: N/A")
            self.pc3_label.config(text=f"IR: {ir_val:.2f}")
        else:
            self.update_status("Classifying (PCA)...")
            f_std = (self.features - self.pca_mean) / self.pca_std
            self.pca_values = np.dot(f_std, self.pca_components.T)
            self.update_pca_display()
            self.classification_result = self.classify_nearest_centroid()

        self.update_result_display()
        self.update_status("Classification complete")

    def calculate_yellow_score(self, img_rgb, mask=None):
        """Method: Min(R, G) - B"""
        mean = cv2.mean(img_rgb, mask=mask)[:3] if mask is not None else cv2.mean(img_rgb)[:3]
        return max(0.0, min(mean[0], mean[1]) - mean[2])

    def calculate_black_score(self, img_rgb, mask=None):
        """
        [NEW] Calculate Black Score using Inverted Luminance.
        Formula: 255 - (0.299R + 0.587G + 0.114B)
        High value = Black/Dark. Low value = White/Bright.
        """
        if mask is not None:
            mean = cv2.mean(img_rgb, mask=mask)[:3]
        else:
            mean = cv2.mean(img_rgb)[:3]
            
        r, g, b = mean
        luminance = (0.299 * r) + (0.587 * g) + (0.114 * b)
        return max(0.0, 255 - luminance)

    def extract_features(self):
        if self.current_frame is None or self.segmentation_mask is None: return None
        try:
            if self.camera_available:
                frames = self.pipeline.wait_for_frames(timeout_ms=100)
                ir = np.asanyarray(frames.get_infrared_frame().get_data())
                if ROTATE_PREVIEW: ir = cv2.rotate(ir, ROTATE_DIR)
            else: ir = np.ones(self.current_frame.shape[:2], dtype=np.uint8) * 150

            x1, y1, x2, y2 = self.bbox_coords
            roi_col = self.current_frame[y1:y2, x1:x2]
            roi_ir = ir[y1:y2, x1:x2]
            
            mask = np.zeros(self.current_frame.shape[:2], dtype=np.uint8)
            if self.polygon_coords:
                pts = np.array(self.polygon_coords, np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(mask, [pts], 255)
            else: mask[y1:y2, x1:x2] = 255
            mask_roi = mask[y1:y2, x1:x2]
            roi_bgr = cv2.cvtColor(roi_col, cv2.COLOR_RGB2BGR)

            # --- CUSTOM SCORES ---
            self.current_yellow_score = self.calculate_yellow_score(roi_col, mask_roi)
            self.current_black_score = self.calculate_black_score(roi_col, mask_roi) # [NEW]

            # --- STANDARD FEATURES ---
            feats = []
            mean_bgr = cv2.mean(roi_bgr)[:3]
            feats.append(mean_bgr[2]) # R
            feats.append(mean_bgr[1]) # G
            
            mean_poly = cv2.mean(roi_bgr, mask=mask_roi)[:3]
            feats.append(mean_poly[0]) # B
            
            pix = roi_bgr[mask_roi > 0]
            feats.append(np.std(pix, axis=0)[0] if len(pix) > 0 else 0.0) # B std
            
            hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
            h_mean = cv2.mean(hsv)[:3]
            feats.append(h_mean[0]) # H
            feats.append(h_mean[1]) # S
            
            h_poly = cv2.mean(hsv, mask=mask_roi)[:3]
            feats.append(h_poly[0]) # H
            feats.append(h_poly[1]) # S
            feats.append(h_poly[2]) # V
            
            gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist /= (hist.sum() + 1e-7)
            feats.append(-np.sum(hist * np.log2(hist + 1e-7))) # Entropy bbox
            
            hist_p = cv2.calcHist([gray], [0], mask_roi, [256], [0, 256])
            hist_p /= (hist_p.sum() + 1e-7)
            feats.append(-np.sum(hist_p * np.log2(hist_p + 1e-7))) # Entropy poly
            
            feats.append(cv2.mean(roi_ir, mask=mask_roi)[0]) # IR
            
            return np.array(feats)
        except Exception as e:
            print(f"Feature error: {e}")
            return None

    def classify_by_ir_cluster(self, ir_value):
        dists = {k: abs(ir_value - v) for k, v in self.ir_thresholds.items()}
        pred = min(dists, key=dists.get)
        return {'class': pred, 'distance': dists[pred], 'distances': dists, 
                'usable': pred in {'Light oil', 'Medium oil', 'Dark oil'},
                'ir_value': ir_value, 'threshold': self.ir_thresholds[pred]}

    def classify_nearest_centroid(self):
        pt = self.pca_values
        dists = {k: np.linalg.norm(pt - v) for k, v in self.centroids.items()}
        pred = min(dists, key=dists.get)
        return {'class': pred, 'distance': dists[pred], 'distances': dists, 
                'usable': pred in self.usable_classes}

    def update_pca_display(self):
        v = self.pca_values
        self.pc1_label.config(text=f"PC1: {v[0]:+.3f}")
        self.pc2_label.config(text=f"PC2: {v[1]:+.3f}")
        self.pc3_label.config(text=f"PC3: {v[2]:+.3f}")

    def update_result_display(self):
        res = self.classification_result
        self.result_label.config(text=f"Oil Type: {res['class']}")
        
        bg = "#C8E6C9" if res['usable'] else "#FFCDD2"
        self.usability_label.config(text="USABLE OIL" if res['usable'] else "UNUSABLE OIL", background=bg)
        self.confidence_label.config(text=f"Confidence: {100*np.exp(-res['distance']):.1f}%")

        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, "=== Classification Details ===\n\n")
        self.info_text.insert(tk.END, f"Class: {res['class']}\n")
        
        # [NEW] Display Yellow AND Black scores
        self.info_text.insert(tk.END, f"Yellow Score: {self.current_yellow_score:.2f} (Max 255)\n")
        self.info_text.insert(tk.END, f"Black Score : {self.current_black_score:.2f} (Max 255)\n")
        self.info_text.insert(tk.END, f"---------------------------\n")

        if self.classification_method.get() == "ir":
            self.info_text.insert(tk.END, f"IR Value: {res['ir_value']:.2f}\nDiff: {res['distance']:.2f}\n\n")
            sorted_d = sorted(res['distances'].items(), key=lambda x: x[1])
            for k, v in sorted_d:
                m = " <--" if k == res['class'] else ""
                self.info_text.insert(tk.END, f"{k:15s}: {v:6.2f}{m}\n")
        else:
            self.info_text.insert(tk.END, f"Dist: {res['distance']:.3f}\n\n")
            sorted_d = sorted(res['distances'].items(), key=lambda x: x[1])
            for k, v in sorted_d:
                m = " <--" if k == res['class'] else ""
                self.info_text.insert(tk.END, f"{k:15s}: {v:6.3f}{m}\n")
                
        self.info_text.insert(tk.END, f"\n=== Features ===\n")
        for n, v in zip(self.feature_columns, self.features):
            self.info_text.insert(tk.END, f"{n:15s}: {v:8.3f}\n")

    def load_image(self):
        fn = filedialog.askopenfilename()
        if fn:
            img = cv2.imread(fn)
            self.current_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.display_image(self.current_frame)
            self.segmentation_mask = None
            self.update_status(f"Loaded: {fn}")

    def take_snapshot(self):
        if self.current_frame is not None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"snapshot_{ts}.jpg", cv2.cvtColor(self.current_frame, cv2.COLOR_RGB2BGR))
            self.update_status(f"Saved snapshot_{ts}.jpg")

    def update_status(self, msg):
        self.status_label.config(text=msg)
        self.root.update_idletasks()

    def cleanup(self):
        if self.camera_available: self.pipeline.stop()

def main():
    root = tk.Tk()
    app = OilClassificationGUI(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.cleanup(), root.destroy()))
    root.mainloop()

if __name__ == "__main__":
    main()