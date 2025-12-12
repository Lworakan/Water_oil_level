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

        # Auto-segmentation variables
        self.auto_seg_results = None
        self.current_clusters = {}

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
        ttk.Button(button_frame, text="Cluster Region",
                  command=self.run_clustering_on_mask).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Yellow Liquid Detect",
                  command=self.detect_yellow_liquid,
                  style="Warning.TButton").pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Load Image",
                  command=self.load_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Take Snapshot",
                  command=self.take_snapshot).pack(side=tk.LEFT, padx=5)

        # Cluster visibility controls
        cluster_vis_frame = ttk.LabelFrame(left_frame, text="Cluster Display", padding="10")
        cluster_vis_frame.pack(pady=5, fill=tk.X)

        # Visibility checkboxes
        self.show_cluster_0 = tk.BooleanVar(value=True)
        self.show_cluster_1 = tk.BooleanVar(value=True)

        ttk.Checkbutton(cluster_vis_frame, text="Show Cluster 0 (Green)",
                       variable=self.show_cluster_0).pack(anchor=tk.W)
        ttk.Checkbutton(cluster_vis_frame, text="Show Cluster 1 (Magenta)",
                       variable=self.show_cluster_1).pack(anchor=tk.W)

        # Cluster selection for classification
        cluster_select_frame = ttk.LabelFrame(left_frame, text="Select Cluster for Classification", padding="10")
        cluster_select_frame.pack(pady=5, fill=tk.X)

        self.selected_cluster_for_classification = tk.StringVar(value="auto")

        ttk.Radiobutton(cluster_select_frame, text="Auto (Larger Area)",
                       variable=self.selected_cluster_for_classification,
                       value="auto").pack(anchor=tk.W)
        ttk.Radiobutton(cluster_select_frame, text="Cluster 0 (Green)",
                       variable=self.selected_cluster_for_classification,
                       value="cluster_0").pack(anchor=tk.W)
        ttk.Radiobutton(cluster_select_frame, text="Cluster 1 (Magenta)",
                       variable=self.selected_cluster_for_classification,
                       value="cluster_1").pack(anchor=tk.W)

        # Cluster info labels
        self.cluster_0_info = ttk.Label(cluster_select_frame, text="Cluster 0: N/A",
                                        font=("Consolas", 9), foreground="green")
        self.cluster_0_info.pack(anchor=tk.W, pady=(5,0))

        self.cluster_1_info = ttk.Label(cluster_select_frame, text="Cluster 1: N/A",
                                        font=("Consolas", 9), foreground="purple")
        self.cluster_1_info.pack(anchor=tk.W)

        # Classification method selection
        method_frame = ttk.LabelFrame(left_frame, text="Classification Method", padding="10")
        method_frame.pack(pady=5, fill=tk.X)

        self.classification_method = tk.StringVar(value="ir")

        ttk.Radiobutton(method_frame, text="IR Cluster (Recommended)",
                       variable=self.classification_method,
                       value="ir").pack(anchor=tk.W)
        ttk.Radiobutton(method_frame, text="PCA Model",
                       variable=self.classification_method,
                       value="pca").pack(anchor=tk.W)

        # Classify button (moved here after cluster selection)
        ttk.Button(cluster_select_frame, text="🔍 Classify Selected Cluster",
                  command=self.run_classification).pack(pady=10, fill=tk.X)

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
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Apply rotation if configured
                if ROTATE_PREVIEW:
                    frame_rgb = cv2.rotate(frame_rgb, ROTATE_DIR)

                self.current_frame = frame_rgb

                # Draw cluster visualization if available
                display_frame = self.current_frame.copy()
                if self.current_clusters:
                    # Draw cluster 0 (main bottle region) - only if visibility is enabled
                    if self.show_cluster_0.get():
                        cluster_0 = self.current_clusters.get('cluster_0', {})
                        if cluster_0.get('mask') is not None:
                            mask_0 = cluster_0['mask']
                            contours, _ = cv2.findContours(mask_0.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            cv2.drawContours(display_frame, contours, -1, (0, 255, 0), 2)  # Green

                            # Draw bounding box for cluster 0
                            if cluster_0['bbox'] != (0, 0, 0, 0):
                                x1, y1, x2, y2 = cluster_0['bbox']
                                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Draw cluster 1 (secondary region) - only if visibility is enabled
                    if self.show_cluster_1.get():
                        cluster_1 = self.current_clusters.get('cluster_1', {})
                        if cluster_1.get('mask') is not None:
                            mask_1 = cluster_1['mask']
                            contours, _ = cv2.findContours(mask_1.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            cv2.drawContours(display_frame, contours, -1, (255, 0, 255), 2)  # Magenta
                elif self.segmentation_mask is not None:
                    # Fallback to basic segmentation display
                    overlay = np.zeros_like(display_frame)
                    overlay[self.segmentation_mask] = [0, 255, 0]
                    display_frame = cv2.addWeighted(display_frame, 0.7, overlay, 0.3, 0)

                    if self.bbox_coords is not None:
                        x1, y1, x2, y2 = self.bbox_coords
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

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
        # Adjust thumbnail size based on rotation
        if ROTATE_PREVIEW:
            image_pil.thumbnail((360, 480))  # Rotated dimensions
        else:
            image_pil.thumbnail((640, 480))
        photo = ImageTk.PhotoImage(image_pil)
        self.video_label.config(image=photo)
        self.video_label.image = photo

    def run_segmentation(self):
        """Run SAM3 automatic segmentation with clustering"""
        if self.current_frame is None:
            messagebox.showwarning("Warning", "No image available")
            return

        self.update_status("Running auto-segmentation with clustering...")

        try:
            # Convert RGB to BGR for auto_segmenter
            image_bgr = cv2.cvtColor(self.current_frame, cv2.COLOR_RGB2BGR)

            # Process frame with AutoSegmenter (includes SAM3 + K-means clustering)
            self.auto_seg_results = self.auto_segmenter.process_frame(
                image_bgr, prompt="bottle", n_clusters=2
            )

            # Update cluster data
            self.update_cluster_data()

            # Get combined segmentation mask from clustering
            if self.auto_seg_results and self.auto_seg_results['segmentation_mask'] is not None:
                # Choose the cluster with the larger area
                cluster_0 = self.current_clusters.get('cluster_0', {})
                cluster_1 = self.current_clusters.get('cluster_1', {})

                # Calculate areas
                area_0 = np.sum(cluster_0.get('mask', 0)) if cluster_0.get('mask') is not None else 0
                area_1 = np.sum(cluster_1.get('mask', 0)) if cluster_1.get('mask') is not None else 0

                # Select larger cluster
                if area_0 >= area_1 and area_0 > 0:
                    selected_cluster = cluster_0
                    selected_cluster_name = 'cluster_0'
                    print(f"Selected cluster_0 (area: {area_0} pixels)")
                elif area_1 > 0:
                    selected_cluster = cluster_1
                    selected_cluster_name = 'cluster_1'
                    print(f"Selected cluster_1 (area: {area_1} pixels)")
                else:
                    selected_cluster = None
                    selected_cluster_name = None

                if selected_cluster is not None and selected_cluster.get('mask') is not None:
                    self.segmentation_mask = selected_cluster['mask']
                    self.bbox_coords = selected_cluster['bbox']

                    # Get polygon from selected cluster
                    polygons = self.auto_seg_results.get('polygons', {})
                    selected_polygon = polygons.get(selected_cluster_name, [])
                    if selected_polygon:
                        self.polygon_coords = selected_polygon
                    else:
                        self.polygon_coords = None

                    n_clusters = self.auto_seg_results['clustering']['n_clusters']
                    self.update_status(f"Segmentation complete ({n_clusters} clusters, using {selected_cluster_name})")
                else:
                    messagebox.showwarning("Warning", "No valid segmentation found")
                    self.update_status("Segmentation failed: empty mask")
            else:
                messagebox.showwarning("Warning", "Segmentation failed")
                self.update_status("Segmentation failed: no objects detected")

        except Exception as e:
            messagebox.showerror("Error", f"Segmentation failed: {e}")
            self.update_status("Segmentation failed")
            import traceback
            traceback.print_exc()

    def run_clustering_on_mask(self):
        """Run clustering on the current segmentation mask"""
        if self.current_frame is None:
            messagebox.showwarning("Warning", "No image available")
            return
            
        if self.segmentation_mask is None:
            messagebox.showwarning("Warning", "No segmentation mask available. Run segmentation first.")
            return

        self.update_status("Running clustering on current mask...")

        try:
            # Convert RGB to BGR for auto_segmenter
            image_bgr = cv2.cvtColor(self.current_frame, cv2.COLOR_RGB2BGR)
            
            # Use the existing mask
            mask = self.segmentation_mask
            
            # Apply clustering (force 2 clusters)
            clustering_result = self.auto_segmenter.apply_kmeans_clustering(
                image_bgr, mask, n_clusters=2
            )
            
            # Get geometric features
            bboxes = self.auto_segmenter.get_cluster_bboxes(clustering_result['cluster_masks'])
            polygons = self.auto_segmenter.get_cluster_polygons(clustering_result['cluster_masks'])
            
            # Create visualization
            visualization = self.auto_segmenter.visualize_clustering(image_bgr, clustering_result)
            
            # Update results - we keep the original segmentation mask but update clustering details
            self.auto_seg_results = {
                'segmentation_mask': mask,
                'clustering': clustering_result,
                'bboxes': bboxes,
                'polygons': polygons,
                'visualization': visualization
            }
            
            # Update UI
            self.update_cluster_data()
            
            # Select larger cluster by default
            cluster_0 = self.current_clusters.get('cluster_0', {})
            cluster_1 = self.current_clusters.get('cluster_1', {})
            
            area_0 = np.sum(cluster_0.get('mask', 0)) if cluster_0.get('mask') is not None else 0
            area_1 = np.sum(cluster_1.get('mask', 0)) if cluster_1.get('mask') is not None else 0
            
            if area_0 >= area_1 and area_0 > 0:
                selected_cluster = cluster_0
                selected_cluster_name = 'cluster_0'
            elif area_1 > 0:
                selected_cluster = cluster_1
                selected_cluster_name = 'cluster_1'
            else:
                selected_cluster = None
                selected_cluster_name = None

            if selected_cluster is not None and selected_cluster.get('mask') is not None:
                self.segmentation_mask = selected_cluster['mask']
                self.bbox_coords = selected_cluster['bbox']
                
                # Get polygon from selected cluster
                polygons = self.auto_seg_results.get('polygons', {})
                selected_polygon = polygons.get(selected_cluster_name, [])
                if selected_polygon:
                    self.polygon_coords = selected_polygon
                else:
                    self.polygon_coords = None
                    
                self.update_status(f"Clustering complete. Using {selected_cluster_name}")
            else:
                self.update_status("Clustering complete (no valid clusters)")

        except Exception as e:
            messagebox.showerror("Error", f"Clustering failed: {e}")
            self.update_status("Clustering failed")
            import traceback
            traceback.print_exc()

    def detect_yellow_liquid(self):
        """Detect yellow liquid using SAM3 with specific prompt"""
        if self.current_frame is None:
            messagebox.showwarning("Warning", "No image available")
            return

        self.update_status("Detecting yellow liquid...")

        try:
            # Convert RGB to BGR for auto_segmenter
            image_bgr = cv2.cvtColor(self.current_frame, cv2.COLOR_RGB2BGR)

            # Process frame with AutoSegmenter using "yellow liquid" prompt
            yellow_seg_results = self.auto_segmenter.process_frame(
                image_bgr, prompt="yellow liquid", n_clusters=2
            )

            # Check if any segmentation was found
            if yellow_seg_results and yellow_seg_results['segmentation_mask'] is not None:
                total_area = np.sum(yellow_seg_results['segmentation_mask'])

                if total_area > 100:  # Minimum area threshold (pixels)
                    # Yellow liquid detected - use the results
                    self.auto_seg_results = yellow_seg_results
                    self.update_cluster_data()

                    # Select largest cluster
                    cluster_0 = self.current_clusters.get('cluster_0', {})
                    cluster_1 = self.current_clusters.get('cluster_1', {})

                    area_0 = np.sum(cluster_0.get('mask', 0)) if cluster_0.get('mask') is not None else 0
                    area_1 = np.sum(cluster_1.get('mask', 0)) if cluster_1.get('mask') is not None else 0

                    if area_0 >= area_1 and area_0 > 0:
                        selected_cluster = cluster_0
                        selected_cluster_name = 'cluster_0'
                    elif area_1 > 0:
                        selected_cluster = cluster_1
                        selected_cluster_name = 'cluster_1'
                    else:
                        selected_cluster = None

                    if selected_cluster:
                        self.segmentation_mask = selected_cluster['mask']
                        self.bbox_coords = selected_cluster['bbox']

                        polygons = yellow_seg_results.get('polygons', {})
                        selected_polygon = polygons.get(selected_cluster_name, [])
                        if selected_polygon:
                            self.polygon_coords = selected_polygon
                        else:
                            self.polygon_coords = None

                        self.update_status(f"✓ Yellow liquid detected! ({total_area} pixels)")
                        messagebox.showinfo("Detection Success",
                                          f"Yellow liquid detected!\nArea: {total_area} pixels\nUsing {selected_cluster_name}")
                    else:
                        messagebox.showwarning("Detection Failed",
                                             "Yellow liquid area too small or invalid.\nNot acceptable.")
                        self.update_status("✗ Yellow liquid detection failed: area too small")
                else:
                    # Area too small
                    messagebox.showwarning("Detection Failed",
                                         f"Yellow liquid area too small ({total_area} pixels).\nNot acceptable.")
                    self.update_status(f"✗ Yellow liquid not detected (area: {total_area} px)")
            else:
                # No segmentation found
                messagebox.showerror("Detection Failed",
                                   "No yellow liquid detected in the image.\nNot acceptable.")
                self.update_status("✗ No yellow liquid detected")

        except Exception as e:
            messagebox.showerror("Error", f"Yellow liquid detection failed: {e}")
            self.update_status("Yellow liquid detection error")
            import traceback
            traceback.print_exc()

    def update_cluster_data(self):
        """Update cluster data from auto-segmentation results"""
        if not self.auto_seg_results:
            return

        clustering = self.auto_seg_results['clustering']
        bboxes = self.auto_seg_results['bboxes']

        # Update stored cluster data
        self.current_clusters = {
            'cluster_0': {
                'bbox': bboxes.get('cluster_0', (0, 0, 0, 0)),
                'mask': clustering['cluster_masks'].get('cluster_0'),
                'stats': clustering['cluster_stats'].get('cluster_0', {})
            },
            'cluster_1': {
                'bbox': bboxes.get('cluster_1', (0, 0, 0, 0)),
                'mask': clustering['cluster_masks'].get('cluster_1'),
                'stats': clustering['cluster_stats'].get('cluster_1', {})
            }
        }

        # Update cluster info labels
        cluster_0 = self.current_clusters.get('cluster_0', {})
        cluster_1 = self.current_clusters.get('cluster_1', {})

        area_0 = np.sum(cluster_0.get('mask', 0)) if cluster_0.get('mask') is not None else 0
        area_1 = np.sum(cluster_1.get('mask', 0)) if cluster_1.get('mask') is not None else 0

        stats_0 = cluster_0.get('stats', {})
        stats_1 = cluster_1.get('stats', {})

        pct_0 = stats_0.get('percentage', 0)
        pct_1 = stats_1.get('percentage', 0)

        self.cluster_0_info.config(text=f"Cluster 0: {area_0} px ({pct_0:.1f}%)")
        self.cluster_1_info.config(text=f"Cluster 1: {area_1} px ({pct_1:.1f}%)")

    def run_classification(self):
        """Run feature extraction, PCA, and classification"""
        # Determine which cluster to use based on selection
        selection = self.selected_cluster_for_classification.get()

        if selection == "auto":
            # Use automatic selection (already set in run_segmentation)
            if self.segmentation_mask is None:
                messagebox.showwarning("Warning", "Please run segmentation first")
                return
            self.update_status("Extracting features from auto-selected cluster...")
        elif selection == "cluster_0":
            # Force use of cluster 0
            cluster_0 = self.current_clusters.get('cluster_0', {})
            if cluster_0.get('mask') is None or np.sum(cluster_0.get('mask', 0)) == 0:
                messagebox.showwarning("Warning", "Cluster 0 not available. Run segmentation first.")
                return
            self.segmentation_mask = cluster_0['mask']
            self.bbox_coords = cluster_0['bbox']
            polygons = self.auto_seg_results.get('polygons', {}) if self.auto_seg_results else {}
            self.polygon_coords = polygons.get('cluster_0', [])
            self.update_status("Extracting features from Cluster 0...")
        elif selection == "cluster_1":
            # Force use of cluster 1
            cluster_1 = self.current_clusters.get('cluster_1', {})
            if cluster_1.get('mask') is None or np.sum(cluster_1.get('mask', 0)) == 0:
                messagebox.showwarning("Warning", "Cluster 1 not available. Run segmentation first.")
                return
            self.segmentation_mask = cluster_1['mask']
            self.bbox_coords = cluster_1['bbox']
            polygons = self.auto_seg_results.get('polygons', {}) if self.auto_seg_results else {}
            self.polygon_coords = polygons.get('cluster_1', [])
            self.update_status("Extracting features from Cluster 1...")

        # Extract features
        self.features = self.extract_features()
        if self.features is None:
            messagebox.showerror("Error", "Feature extraction failed")
            return

        # Get classification method
        method = self.classification_method.get()

        if method == "ir":
            # IR Cluster-based classification
            self.update_status("Classifying using IR cluster...")

            # Feature 12 is the IR cluster value (index 11)
            ir_value = self.features[11]

            # Classify by IR
            self.classification_result = self.classify_by_ir_cluster(ir_value)

            # Set dummy PCA values for display
            self.pca_values = np.array([0.0, 0.0, ir_value])
            self.pc1_label.config(text="PC1: N/A (IR mode)")
            self.pc2_label.config(text="PC2: N/A (IR mode)")
            self.pc3_label.config(text=f"IR Value: {ir_value:.2f}")

        else:  # method == "pca"
            # PCA-based classification
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

                # Apply rotation if configured
                if ROTATE_PREVIEW:
                    img_ir = cv2.rotate(img_ir, ROTATE_DIR)
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

            # 5-6. hsv_bbox_mean_h, hsv_bbox_mean_s (NO V for bbox!)
            hsv_bbox = cv2.cvtColor(roi_color_bgr, cv2.COLOR_BGR2HSV)
            hsv_bbox_mean = cv2.mean(hsv_bbox)[:3]
            features.append(hsv_bbox_mean[0])  # H
            features.append(hsv_bbox_mean[1])  # S (skip V)

            # 7-9. hsv_poly_mean_h, hsv_poly_mean_s, hsv_poly_mean_v
            hsv_poly_mean = cv2.mean(hsv_bbox, mask=mask_roi)[:3]
            features.append(hsv_poly_mean[0])  # H
            features.append(hsv_poly_mean[1])  # S
            features.append(hsv_poly_mean[2])  # V

            # 10. color_bbox_entropy
            gray_bbox = cv2.cvtColor(roi_color_bgr, cv2.COLOR_BGR2GRAY)
            hist_bbox = cv2.calcHist([gray_bbox], [0], None, [256], [0, 256])
            hist_bbox = hist_bbox / (hist_bbox.sum() + 1e-7)
            entropy_bbox = -np.sum(hist_bbox * np.log2(hist_bbox + 1e-7))
            features.append(entropy_bbox)

            # 11. color_poly_entropy
            hist_poly = cv2.calcHist([gray_bbox], [0], mask_roi, [256], [0, 256])
            hist_poly = hist_poly / (hist_poly.sum() + 1e-7)
            entropy_poly = -np.sum(hist_poly * np.log2(hist_poly + 1e-7))
            features.append(entropy_poly)

            # 12. ir_cluster (mean IR value)
            ir_mean = cv2.mean(roi_ir, mask=mask_roi)[0]
            features.append(ir_mean)

            return np.array(features)

        except Exception as e:
            print(f"Feature extraction error: {e}")
            return None

    def classify_by_ir_cluster(self, ir_value):
        """Classify using IR cluster value (nearest threshold)"""
        distances = {}

        # Calculate distance to each oil type's IR threshold
        for oil_type, threshold in self.ir_thresholds.items():
            distance = abs(ir_value - threshold)
            distances[oil_type] = distance

        # Find nearest threshold
        predicted_class = min(distances, key=distances.get)
        min_distance = distances[predicted_class]

        # Define usable classes for IR method
        usable_ir_classes = {'Light oil', 'Medium oil', 'Dark oil'}

        return {
            'class': predicted_class,
            'distance': min_distance,
            'distances': distances,
            'usable': predicted_class in usable_ir_classes,
            'ir_value': ir_value,
            'threshold': self.ir_thresholds[predicted_class]
        }

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
        method = self.classification_method.get()

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
        self.info_text.insert(tk.END, f"Method: {'IR Cluster' if method == 'ir' else 'PCA Model'}\n")
        self.info_text.insert(tk.END, f"Predicted Class: {result['class']}\n")

        if method == "ir":
            # IR-specific information
            ir_value = result.get('ir_value', 0)
            threshold = result.get('threshold', 0)
            self.info_text.insert(tk.END, f"IR Value: {ir_value:.2f}\n")
            self.info_text.insert(tk.END, f"Threshold: {threshold:.2f}\n")
            self.info_text.insert(tk.END, f"Difference: {result['distance']:.2f}\n\n")

            self.info_text.insert(tk.END, "=== IR Thresholds Distance ===\n")
            sorted_distances = sorted(result['distances'].items(), key=lambda x: x[1])
            for oil_type, dist in sorted_distances:
                marker = " <--" if oil_type == result['class'] else ""
                threshold_val = self.ir_thresholds[oil_type]
                self.info_text.insert(tk.END, f"{oil_type:18s} ({threshold_val:6.2f}): {dist:6.2f}{marker}\n")
        else:
            # PCA-specific information
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
