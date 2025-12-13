"""
Enhanced Oil Quality Classification GUI with 2D/3D Classification
Implements IR + Yellow + Black score thresholds for improved accuracy
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
import threading

# --- CONFIGURATION ---
ROTATE_PREVIEW = True
ROTATE_DIR = cv2.ROTATE_90_COUNTERCLOCKWISE
MIN_CAPACITY_PIXELS = 20200  # Minimum pixels for capacity check

class AutomationState:
    """Track the automation sequence state"""
    IDLE = "idle"
    LIQUID_SEGMENT = "liquid_segment"
    CLUSTER_CHECK = "cluster_check"
    WAITING_IR_READ = "waiting_ir_read"
    IR_DIFFERENCE_CHECK = "ir_difference_check"
    SINGLE_LIQUID_MIDDLE = "single_liquid_middle"
    WAITING_RGB_SINGLE = "waiting_rgb_single"
    DUAL_LIQUID_CLUSTER1 = "dual_liquid_cluster1"
    WAITING_RGB_CLUSTER1 = "waiting_rgb_cluster1"
    DUAL_LIQUID_CLUSTER2 = "dual_liquid_cluster2"
    WAITING_RGB_CLUSTER2 = "waiting_rgb_cluster2"
    ANALYSIS_COMPLETE = "analysis_complete"

class OilClassificationEnhancedGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Enhanced Oil Quality Classification System (2D/3D)")
        self.root.geometry("1600x1000")

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
        self.current_black_score = 0.0

        # Auto-segmentation variables
        self.auto_seg_results = None
        self.current_clusters = {}
        
        # Automation state variables
        self.automation_state = AutomationState.IDLE
        self.cluster1_data = None
        self.cluster2_data = None
        self.ir_values = {}
        self.rgb_values = {}
        self.final_results = {}

        # Display variables
        self.display_scale = 1.0

        # Processing flag to control video updates during heavy computation
        self.is_processing = False

        # Cached overlay data to prevent recomputation every frame
        self.cached_overlay = None
        self.overlay_needs_update = False

        # Enhanced 2D/3D thresholds for each oil type
        self.oil_thresholds = {
            'Water': {
                'ir': 100.40,
                'yellow': 2.0,      # Very low yellow (clear/transparent)
                'black': 20.0       # Very low black (bright/light)
            },
            'Light oil': {
                'ir':200.98,
                'yellow': 98.0,     # Higher yellow (golden color)
                'black': 126.0       # Lower black (lighter appearance)
            },
            'Medium oil': {
                'ir': 145.0,
                'yellow': 145.0,     # Medium yellow
                'black': 89.0       # Medium black
            },
            'Dark oil': {
                'ir': 111.0,
                'yellow': 7.3,     # Lower yellow (less golden)
                'black': 217.0      # High black (darker appearance)
            },
            'Motor oil': {
                'ir': 35.0,
                'yellow': 0.0,     # Low yellow
                'black': 225.0      # High black (dark, viscous)
            },
            'Lard': {
                'ir': 60.0,
                'yellow': 20.0,     # Higher yellow (fatty, yellowish)
                'black': 160.0       # Medium-low black (relatively light)
            },
            'Mix (Oil+Motor)': {
                'ir': 47.12,
                'yellow': 22.0,     # Low yellow
                'black': 85.0       # High black
            },
            'Mix (Oil+Lard)': {
                'ir': 91.52,
                'yellow': 40.0,     # Medium yellow
                'black': 55.0       # Medium black
            }
        }
        
        # Weights for distance calculation (adjust based on importance)
        self.classification_weights = {
            'ir': 1.0,
            'yellow': 0.8,
            'black': 0.8
        }
        
        # Normalization factors for balanced distance calculation
        self.normalization_factors = {
            'ir': 200.0,        # Max expected IR range
            'yellow': 100.0,    # Max expected yellow score
            'black': 150.0      # Max expected black score
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

        # Initialize button states
        self.update_button_states()

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
            print("⚠ PCA model file not found, using threshold-based classification only")
            self.pca_mean = None

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

    def calculate_3d_distance(self, ir_value, yellow_score, black_score, oil_type):
        """Calculate normalized 3D distance to oil type thresholds"""
        thresholds = self.oil_thresholds[oil_type]
        
        # Normalize each dimension
        ir_diff = abs(ir_value - thresholds['ir']) / self.normalization_factors['ir']
        yellow_diff = abs(yellow_score - thresholds['yellow']) / self.normalization_factors['yellow']
        black_diff = abs(black_score - thresholds['black']) / self.normalization_factors['black']
        
        # Apply weights
        weighted_ir = ir_diff * self.classification_weights['ir']
        weighted_yellow = yellow_diff * self.classification_weights['yellow']
        weighted_black = black_diff * self.classification_weights['black']
        
        # Calculate weighted Euclidean distance
        distance = np.sqrt(weighted_ir**2 + weighted_yellow**2 + weighted_black**2)
        
        return distance

    def classify_by_3d_thresholds(self, ir_value, yellow_score, black_score):
        """Enhanced classification using 3D distance to thresholds"""
        distances = {}
        
        # Calculate distance to each oil type
        for oil_type in self.oil_thresholds.keys():
            distance = self.calculate_3d_distance(ir_value, yellow_score, black_score, oil_type)
            distances[oil_type] = distance
        
        # Find closest match
        best_match = min(distances, key=distances.get)
        confidence = 1.0 / (1.0 + distances[best_match])  # Convert distance to confidence
        
        # Calculate individual component distances for analysis
        best_thresholds = self.oil_thresholds[best_match]
        component_distances = {
            'ir': abs(ir_value - best_thresholds['ir']),
            'yellow': abs(yellow_score - best_thresholds['yellow']),
            'black': abs(black_score - best_thresholds['black'])
        }
        
        result = {
            'class': best_match,
            'confidence': confidence,
            'total_distance': distances[best_match],
            'all_distances': distances,
            'component_distances': component_distances,
            'usable': best_match in {'Light oil', 'Medium oil', 'Dark oil'},
            'ir_value': ir_value,
            'yellow_score': yellow_score,
            'black_score': black_score,
            'thresholds': best_thresholds
        }
        
        return result

    def classify_by_ir_and_rgb(self, ir_value, yellow_score, black_score):
        """Main classification function using enhanced 3D approach"""
        return self.classify_by_3d_thresholds(ir_value, yellow_score, black_score)

    def create_widgets(self):
        """Create GUI widgets with enhanced display"""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Left panel - Video feed
        left_frame = ttk.LabelFrame(main_frame, text="Camera Feed", padding="10")
        left_frame.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.video_label = ttk.Label(left_frame)
        self.video_label.pack()

        # Automation Control Panel
        auto_frame = ttk.LabelFrame(left_frame, text="Automation Control", padding="10")
        auto_frame.pack(pady=5, fill=tk.X)

        # Separate buttons for each phase
        buttons_frame = ttk.Frame(auto_frame)
        buttons_frame.pack(pady=5)

        # 1. Start Analysis button
        self.start_button = ttk.Button(buttons_frame, text="🚀 START ANALYSIS",
                                       command=self.start_analysis,
                                       width=20)
        self.start_button.grid(row=0, column=0, padx=2, pady=2)

        # 2. IR Read button
        self.ir_button = ttk.Button(buttons_frame, text="📡 IR READ",
                                    command=self.read_ir_manual,
                                    width=20, state="disabled")
        self.ir_button.grid(row=0, column=1, padx=2, pady=2)

        # 3. RGB Read button
        self.rgb_button = ttk.Button(buttons_frame, text="🎨 RGB READ",
                                     command=self.read_rgb_manual,
                                     width=20, state="disabled")
        self.rgb_button.grid(row=1, column=0, padx=2, pady=2)

        # 4. Reset button
        self.reset_button = ttk.Button(buttons_frame, text="🔄 RESET",
                                       command=self.reset_automation,
                                       width=20)
        self.reset_button.grid(row=1, column=1, padx=2, pady=2)

        # Current step display
        self.step_label = ttk.Label(auto_frame, text="Ready to start analysis",
                                   font=("Arial", 12, "bold"), foreground="blue")
        self.step_label.pack(pady=5)

        # Classification Method Display
        method_frame = ttk.LabelFrame(left_frame, text="Classification Method", padding="10")
        method_frame.pack(pady=5, fill=tk.X)
        
        method_label = ttk.Label(method_frame, text="Enhanced 3D Classification\n(IR + Yellow + Black)", 
                               font=("Arial", 10, "bold"), foreground="green")
        method_label.pack()

        # Manual controls (for debugging)
        manual_frame = ttk.LabelFrame(left_frame, text="Manual Controls (Debug)", padding="10")
        manual_frame.pack(pady=5, fill=tk.X)

        manual_buttons = ttk.Frame(manual_frame)
        manual_buttons.pack()

        ttk.Button(manual_buttons, text="Load Image", command=self.load_image).pack(side=tk.LEFT, padx=2)
        ttk.Button(manual_buttons, text="Snapshot", command=self.take_snapshot).pack(side=tk.LEFT, padx=2)

        # Middle panel - Current Analysis
        middle_frame = ttk.LabelFrame(main_frame, text="Current Analysis", padding="10")
        middle_frame.grid(row=0, column=1, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Current values with enhanced display (fixed width to prevent wiggling)
        values_frame = ttk.LabelFrame(middle_frame, text="Current Readings", padding="10")
        values_frame.pack(fill=tk.X, pady=5)

        self.ir_label = ttk.Label(values_frame, text="IR: --      ", font=("Courier", 11, "bold"),
                                 background="#E3F2FD", padding=5, width=25, anchor='w')
        self.ir_label.pack(fill=tk.X, pady=2)

        self.yellow_label = ttk.Label(values_frame, text="Yellow Score: --      ", font=("Courier", 11, "bold"),
                                     background="#FFF9C4", padding=5, width=25, anchor='w')
        self.yellow_label.pack(fill=tk.X, pady=2)

        self.black_label = ttk.Label(values_frame, text="Black Score: --      ", font=("Courier", 11, "bold"),
                                    background="#F5F5F5", padding=5, width=25, anchor='w')
        self.black_label.pack(fill=tk.X, pady=2)

        self.pixels_label = ttk.Label(values_frame, text="Pixels: --      ", font=("Courier", 11, "bold"),
                                     background="#E8F5E8", padding=5, width=25, anchor='w')
        self.pixels_label.pack(fill=tk.X, pady=2)

        # Confidence display
        self.confidence_label = ttk.Label(values_frame, text="Confidence: --      ", font=("Courier", 11, "bold"),
                                         background="#F3E5F5", padding=5, width=25, anchor='w')
        self.confidence_label.pack(fill=tk.X, pady=2)

        # Analysis progress
        progress_frame = ttk.LabelFrame(middle_frame, text="Analysis Progress", padding="10")
        progress_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.progress_text = tk.Text(progress_frame, height=15, width=40, font=("Courier", 9), wrap=tk.WORD)
        self.progress_text.pack(fill=tk.BOTH, expand=True)
        progress_scrollbar = ttk.Scrollbar(progress_frame, command=self.progress_text.yview)
        progress_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.progress_text.config(yscrollcommand=progress_scrollbar.set)

        # Right panel - Final Results with Enhanced Display
        right_frame = ttk.LabelFrame(main_frame, text="Final Results", padding="10")
        right_frame.grid(row=0, column=2, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Result display
        result_frame = ttk.LabelFrame(right_frame, text="Classification Result", padding="10")
        result_frame.pack(fill=tk.X, pady=5)

        self.result_label = ttk.Label(result_frame, text="Analysis not started", 
                                     font=("Arial", 14, "bold"), padding=10)
        self.result_label.pack(fill=tk.X)

        self.usability_label = ttk.Label(result_frame, text="", font=("Arial", 12), padding=5)
        self.usability_label.pack(fill=tk.X)

        # Distance comparison display
        distances_frame = ttk.LabelFrame(right_frame, text="Distance Analysis", padding="10")
        distances_frame.pack(fill=tk.X, pady=5)

        self.distances_text = tk.Text(distances_frame, height=8, width=40, font=("Courier", 9))
        self.distances_text.pack(fill=tk.BOTH, expand=True)

        # Detailed results
        details_frame = ttk.LabelFrame(right_frame, text="Detailed Results", padding="10")
        details_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.details_text = tk.Text(details_frame, height=15, width=40, font=("Courier", 9))
        self.details_text.pack(fill=tk.BOTH, expand=True)
        details_scrollbar = ttk.Scrollbar(details_frame, command=self.details_text.yview)
        details_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.details_text.config(yscrollcommand=details_scrollbar.set)

        # Status bar
        self.status_label = ttk.Label(main_frame, text="Ready - Enhanced 3D Classification", relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E))

        # Configure grid weights and minimum sizes to prevent wiggling
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=2, minsize=520)
        main_frame.columnconfigure(1, weight=2, minsize=450)
        main_frame.columnconfigure(2, weight=2, minsize=500)
        main_frame.rowconfigure(0, weight=1)

    def display_distance_analysis(self, result):
        """Display detailed distance analysis"""
        self.distances_text.delete(1.0, tk.END)
        self.distances_text.insert(tk.END, "=== DISTANCE ANALYSIS ===\n")
        self.distances_text.insert(tk.END, f"Best Match: {result['class']}\n")
        self.distances_text.insert(tk.END, f"Confidence: {result['confidence']:.3f}\n\n")
        
        # Show top 3 closest matches
        sorted_distances = sorted(result['all_distances'].items(), key=lambda x: x[1])
        self.distances_text.insert(tk.END, "Top 3 Matches:\n")
        for i, (oil_type, distance) in enumerate(sorted_distances[:3]):
            marker = ">>> " if i == 0 else "    "
            self.distances_text.insert(tk.END, f"{marker}{oil_type}: {distance:.4f}\n")
        
        self.distances_text.insert(tk.END, f"\nComponent Distances:\n")
        comp_dist = result['component_distances']
        self.distances_text.insert(tk.END, f"IR: {comp_dist['ir']:.2f}\n")
        self.distances_text.insert(tk.END, f"Yellow: {comp_dist['yellow']:.2f}\n")
        self.distances_text.insert(tk.END, f"Black: {comp_dist['black']:.2f}\n")

    # [Include all the existing methods from the original code with minimal changes]
    # I'll include the key methods that need modification:

    def update_current_readings(self):
        """Update current reading displays with fixed-width formatting"""
        if self.features is not None and len(self.features) > 11:
            self.ir_label.config(text=f"IR: {self.features[11]:>8.2f}")
        else:
            self.ir_label.config(text="IR: --      ")

        self.yellow_label.config(text=f"Yellow Score: {self.current_yellow_score:>6.2f}")
        self.black_label.config(text=f"Black Score: {self.current_black_score:>7.2f}")

        if self.segmentation_mask is not None:
            pixels = np.sum(self.segmentation_mask)
            self.pixels_label.config(text=f"Pixels: {int(pixels):>8}")
        else:
            self.pixels_label.config(text="Pixels: --      ")

    def show_single_liquid_result(self, result, pixel_count):
        """Display results for single liquid analysis with enhanced info"""
        self.result_label.config(text=f"Type: {result['class']}")

        bg_color = "#C8E6C9" if result['usable'] else "#FFCDD2"
        status_text = "USABLE OIL" if result['usable'] else "UNUSABLE OIL"
        self.usability_label.config(text=status_text, background=bg_color)

        # Update confidence display with fixed width
        self.confidence_label.config(text=f"Confidence: {result['confidence']:>6.3f}")
        
        # Display distance analysis
        self.display_distance_analysis(result)
        
        self.details_text.delete(1.0, tk.END)
        self.details_text.insert(tk.END, "=== ENHANCED SINGLE LIQUID ANALYSIS ===\n\n")
        self.details_text.insert(tk.END, f"Type: {result['class']}\n")
        self.details_text.insert(tk.END, f"Confidence: {result['confidence']:.3f}\n")
        self.details_text.insert(tk.END, f"Total Distance: {result['total_distance']:.4f}\n")
        self.details_text.insert(tk.END, f"Pixels: {pixel_count}\n")
        self.details_text.insert(tk.END, f"Status: {status_text}\n\n")
        
        self.details_text.insert(tk.END, f"=== MEASURED VALUES ===\n")
        self.details_text.insert(tk.END, f"IR Value: {result['ir_value']:.2f}\n")
        self.details_text.insert(tk.END, f"Yellow Score: {result['yellow_score']:.2f}\n")
        self.details_text.insert(tk.END, f"Black Score: {result['black_score']:.2f}\n\n")
        
        self.details_text.insert(tk.END, f"=== TARGET THRESHOLDS ===\n")
        thresh = result['thresholds']
        self.details_text.insert(tk.END, f"IR Threshold: {thresh['ir']:.2f}\n")
        self.details_text.insert(tk.END, f"Yellow Threshold: {thresh['yellow']:.2f}\n")
        self.details_text.insert(tk.END, f"Black Threshold: {thresh['black']:.2f}\n\n")
        
        self.details_text.insert(tk.END, f"=== DISTANCE BREAKDOWN ===\n")
        comp_dist = result['component_distances']
        self.details_text.insert(tk.END, f"IR Distance: {comp_dist['ir']:.2f}\n")
        self.details_text.insert(tk.END, f"Yellow Distance: {comp_dist['yellow']:.2f}\n")
        self.details_text.insert(tk.END, f"Black Distance: {comp_dist['black']:.2f}\n")
        
        self.automation_state = AutomationState.ANALYSIS_COMPLETE
        self.update_button_states()
        self.step_label.config(text="Analysis Complete")

    # [Continue with all other existing methods from the original code]
    # Here I'll add the essential methods that are needed but keep the same logic:

    def update_video_stream(self):
        """Update the video feed with enhanced overlay display"""
        if not self.camera_available: return
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=100)
            color_frame = frames.get_color_frame()
            if color_frame:
                frame = np.asanyarray(color_frame.get_data())
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if ROTATE_PREVIEW: frame_rgb = cv2.rotate(frame_rgb, ROTATE_DIR)
                self.current_frame = frame_rgb

                # Skip heavy overlay rendering when processing
                if self.is_processing:
                    # Just display the raw frame with a processing indicator
                    display_frame = self.current_frame.copy()
                    cv2.putText(display_frame, "PROCESSING...", (20, 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 165, 0), 3)
                    self.display_image(display_frame)
                    self.root.after(30, self.update_video_stream)
                    return

                display_frame = self.current_frame.copy()

                # Render overlays with caching to prevent FPS drops
                # Rebuild cache if overlay needs update
                if self.overlay_needs_update or self.cached_overlay is None:
                    self.cached_overlay = self._build_overlay_cache()
                    self.overlay_needs_update = False

                # Apply cached overlay to current frame
                if self.cached_overlay is not None:
                    display_frame = self._apply_cached_overlay(display_frame, self.cached_overlay)

                self.display_image(display_frame)
        except Exception as e: 
            print(f"Video stream error: {e}")
        self.root.after(30, self.update_video_stream)

    def _build_overlay_cache(self):
        """Build overlay cache to avoid recomputing every frame"""
        try:
            # Case 1: Dual liquid (2 clusters)
            if self.current_clusters:
                cluster_colors = {
                    'cluster_0': (255, 100, 100),  # Light red for cluster 0
                    'cluster_1': (100, 100, 255)   # Light blue for cluster 1
                }

                overlays = []
                for cluster_name, cluster_data in self.current_clusters.items():
                    mask = cluster_data.get('mask')
                    bbox = cluster_data.get('bbox', (0, 0, 0, 0))
                    color = cluster_colors[cluster_name]

                    if mask is not None:
                        # Compute contours once
                        contours, _ = cv2.findContours(mask.astype(np.uint8),
                                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        overlays.append({
                            'type': 'cluster',
                            'mask': mask,
                            'contours': contours,
                            'bbox': bbox,
                            'color': color,
                            'label': cluster_name.upper()
                        })

                # Add 1/3 segment overlay if available
                if self.segmentation_mask is not None and self.bbox_coords:
                    contours, _ = cv2.findContours(self.segmentation_mask.astype(np.uint8),
                                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    overlays.append({
                        'type': 'segment',
                        'mask': self.segmentation_mask,
                        'contours': contours,
                        'bbox': self.bbox_coords,
                        'color': (0, 255, 0),
                        'label': 'MIDDLE 1/3'
                    })

                return {'type': 'dual', 'overlays': overlays}

            # Case 2: Single liquid
            elif self.segmentation_mask is not None:
                # Get full mask for display
                if hasattr(self, 'original_segmentation_mask') and self.original_segmentation_mask is not None:
                    full_mask = self.original_segmentation_mask
                else:
                    full_mask = self.segmentation_mask

                # Compute contours for full mask
                full_contours, _ = cv2.findContours(full_mask.astype(np.uint8),
                                                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Get bbox for full mask
                rows = np.any(full_mask, axis=1)
                cols = np.any(full_mask, axis=0)
                full_bbox = None
                if np.any(rows) and np.any(cols):
                    y_min, y_max = np.where(rows)[0][[0, -1]]
                    x_min, x_max = np.where(cols)[0][[0, -1]]
                    full_bbox = (x_min, y_min, x_max, y_max)

                overlays = [{
                    'type': 'full_liquid',
                    'mask': full_mask,
                    'contours': full_contours,
                    'bbox': full_bbox,
                    'color': (100, 150, 255),
                    'label': 'LIQUID'
                }]

                # Add 1/3 segment if different from full
                if not np.array_equal(self.segmentation_mask, full_mask) and self.bbox_coords:
                    segment_contours, _ = cv2.findContours(self.segmentation_mask.astype(np.uint8),
                                                          cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    overlays.append({
                        'type': 'segment',
                        'mask': self.segmentation_mask,
                        'contours': segment_contours,
                        'bbox': self.bbox_coords,
                        'color': (0, 255, 0),
                        'label': 'MIDDLE 1/3'
                    })

                return {'type': 'single', 'overlays': overlays, 'bbox': self.bbox_coords}

            return None

        except Exception as e:
            print(f"Overlay cache build error: {e}")
            return None

    def _apply_cached_overlay(self, frame, cache):
        """Apply cached overlay to frame (very fast operation - just drawing)"""
        try:
            if cache is None:
                return frame

            overlays = cache.get('overlays', [])

            # Only draw contours and labels - skip heavy blending
            for overlay_data in overlays:
                contours = overlay_data['contours']
                color = overlay_data['color']

                # Draw contours only (very fast)
                cv2.drawContours(frame, contours, -1, color, 2)

                # Draw labels and bboxes
                if overlay_data['type'] == 'cluster':
                    bbox = overlay_data['bbox']
                    if bbox != (0, 0, 0, 0):
                        x1, y1, x2, y2 = bbox
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, overlay_data['label'], (x1, y1-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                elif overlay_data['type'] == 'segment':
                    bbox = overlay_data.get('bbox')
                    if bbox:
                        x1, y1, x2, y2 = bbox
                        cv2.putText(frame, overlay_data['label'], (x1, y2+20),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                elif overlay_data['type'] == 'full_liquid':
                    # Use cached bbox
                    bbox = overlay_data.get('bbox')
                    if bbox:
                        x1, y1 = bbox[0], bbox[1]
                        cv2.putText(frame, overlay_data['label'], (x1, y1-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            return frame

        except Exception as e:
            print(f"Overlay apply error: {e}")
            return frame

    def get_color_description(self, mean_color):
        """Get a descriptive name for the liquid color based on RGB values"""
        r, g, b = mean_color
        
        # Calculate color characteristics
        brightness = (r + g + b) / 3
        yellow_score = min(r, g) - b
        
        if brightness < 80:
            if yellow_score < 5:
                return "Dark/Black"
            else:
                return "Dark Brown"
        elif brightness < 120:
            if yellow_score > 20:
                return "Golden Brown"
            elif b > r and b > g:
                return "Dark Blue"
            else:
                return "Medium Brown"
        elif brightness < 160:
            if yellow_score > 30:
                return "Golden Yellow"
            elif b > r and b > g:
                return "Blue"
            elif g > r and g > b:
                return "Green"
            else:
                return "Light Brown"
        else:  # brightness >= 160
            if yellow_score > 20:
                return "Light Yellow"
            elif b > (r + g) * 0.6:
                return "Light Blue"
            elif abs(r - g) < 20 and abs(g - b) < 20:
                return "Clear/White"
            else:
                return "Light"

    def display_image(self, image):
        """Display image in the GUI"""
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

    def calculate_yellow_score(self, img_rgb, mask=None):
        """Calculate Yellow Score: Min(R, G) - B"""
        if mask is not None:
            mean = cv2.mean(img_rgb, mask=mask.astype(np.uint8))[:3]
        else:
            mean = cv2.mean(img_rgb)[:3]
        return max(0.0, min(mean[0], mean[1]) - mean[2])

    def calculate_black_score(self, img_rgb, mask=None):
        """Calculate Black Score: Inverted Luminance (255 - luminance)"""
        if mask is not None:
            mean = cv2.mean(img_rgb, mask=mask.astype(np.uint8))[:3]
        else:
            mean = cv2.mean(img_rgb)[:3]
            
        r, g, b = mean
        luminance = (0.299 * r) + (0.587 * g) + (0.114 * b)
        return max(0.0, 255 - luminance)

    def load_image(self):
        """Load image from file"""
        fn = filedialog.askopenfilename()
        if fn:
            img = cv2.imread(fn)
            self.current_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.display_image(self.current_frame)
            self.update_status(f"Loaded: {fn}")

    def take_snapshot(self):
        """Take a snapshot of current frame"""
        if self.current_frame is not None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"snapshot_{ts}.jpg", cv2.cvtColor(self.current_frame, cv2.COLOR_RGB2BGR))
            self.update_status(f"Saved snapshot_{ts}.jpg")

    def update_status(self, msg):
        """Update status bar"""
        self.status_label.config(text=msg)
        self.root.update_idletasks()

    def add_progress_log(self, message):
        """Add a message to the progress log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.progress_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.progress_text.see(tk.END)
        self.root.update_idletasks()

    def reset_automation(self):
        """Reset the automation to start state"""
        self.automation_state = AutomationState.IDLE
        self.update_button_states()
        self.step_label.config(text="Ready to start analysis")

        # Clear all data
        self.segmentation_mask = None
        self.original_segmentation_mask = None  # Clear original mask too
        self.bbox_coords = None
        self.current_clusters = {}
        self.ir_values = {}
        self.rgb_values = {}
        self.final_results = {}

        # Clear overlay cache
        self.cached_overlay = None
        self.overlay_needs_update = False
        
        # Clear displays
        self.result_label.config(text="Analysis not started")
        self.usability_label.config(text="", background="")
        self.progress_text.delete(1.0, tk.END)
        self.details_text.delete(1.0, tk.END)
        self.distances_text.delete(1.0, tk.END)
        
        # Reset current readings with fixed width
        self.ir_label.config(text="IR: --      ")
        self.yellow_label.config(text="Yellow Score: --      ")
        self.black_label.config(text="Black Score: --      ")
        self.pixels_label.config(text="Pixels: --      ")
        self.confidence_label.config(text="Confidence: --      ")
        
        self.update_status("Reset complete - ready for new analysis")

    def cleanup(self):
        """Cleanup resources"""
        if self.camera_available: 
            self.pipeline.stop()

    def update_button_states(self):
        """Update button states based on current automation state"""
        # Disable all buttons during processing
        if self.is_processing:
            self.start_button.config(state="disabled")
            self.ir_button.config(state="disabled")
            self.rgb_button.config(state="disabled")
            return

        # Start button - always enabled when not processing
        self.start_button.config(state="normal")

        # IR button - enabled anytime after segmentation is done
        # This allows re-reading IR values at any time
        if self.segmentation_mask is not None:
            self.ir_button.config(state="normal")
        else:
            self.ir_button.config(state="disabled")

        # RGB button - enabled anytime after segmentation is done
        # This allows re-reading RGB values at any time
        if self.segmentation_mask is not None:
            self.rgb_button.config(state="normal")
        else:
            self.rgb_button.config(state="disabled")

    def start_analysis(self):
        """Handler for START ANALYSIS button"""
        try:
            if self.automation_state == AutomationState.IDLE:
                self.start_liquid_segmentation()
            else:
                # Allow re-running segmentation even if already started
                self.add_progress_log("Re-running liquid segmentation...")
                self.start_liquid_segmentation()
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {e}")
            self.add_progress_log(f"ERROR: {e}")

    def read_ir_manual(self):
        """Handler for IR READ button - allows reading IR values anytime"""
        try:
            if self.segmentation_mask is None:
                self.add_progress_log("IR read not available - run analysis first")
                messagebox.showwarning("Warning", "Please run analysis first")
                return

            self.add_progress_log("--- Reading IR values ---")
            self.read_ir_values()
        except Exception as e:
            messagebox.showerror("Error", f"IR read failed: {e}")
            self.add_progress_log(f"ERROR: {e}")

    def read_rgb_manual(self):
        """Handler for RGB READ button - allows reading RGB values anytime"""
        try:
            if self.segmentation_mask is None:
                self.add_progress_log("RGB read not available - run analysis first")
                messagebox.showwarning("Warning", "Please run analysis first")
                return

            self.add_progress_log("--- Reading RGB values ---")

            # Intelligently determine which RGB reading to perform
            if self.automation_state == AutomationState.WAITING_RGB_CLUSTER1:
                # Currently waiting for cluster 0 RGB
                self.read_rgb_cluster1()
            elif self.automation_state == AutomationState.WAITING_RGB_CLUSTER2:
                # Currently waiting for cluster 1 RGB
                self.read_rgb_cluster2()
            elif self.current_clusters:
                # Have clusters - read appropriate cluster based on progress
                if 'cluster_0' not in self.rgb_values:
                    # Start with cluster 0
                    self.add_progress_log("Reading cluster 0 RGB...")
                    self.read_rgb_cluster1()
                elif 'cluster_1' not in self.rgb_values:
                    # Move to cluster 1
                    self.add_progress_log("Reading cluster 1 RGB...")
                    self.read_rgb_cluster2()
                else:
                    # Both done, re-read cluster 0
                    self.add_progress_log("Re-reading cluster 0 RGB...")
                    self.read_rgb_cluster1()
            else:
                # Single liquid
                self.read_rgb_single_liquid()

        except Exception as e:
            messagebox.showerror("Error", f"RGB read failed: {e}")
            self.add_progress_log(f"ERROR: {e}")

    def start_liquid_segmentation(self):
        """Step 1: Start with liquid segmentation"""
        if self.current_frame is None:
            messagebox.showwarning("Warning", "No camera feed available")
            return

        self.add_progress_log("=== STARTING ANALYSIS ===")
        self.add_progress_log("Step 1: Liquid segmentation...")

        # Capture the current frame for processing
        frame_to_process = self.current_frame.copy()

        # Run segmentation in background thread
        def segmentation_worker():
            try:
                self.is_processing = True
                self.root.after(0, self.update_button_states)  # Disable buttons during processing
                self.root.after(0, lambda: self.update_status("Processing segmentation..."))

                image_bgr = cv2.cvtColor(frame_to_process, cv2.COLOR_RGB2BGR)
                self.auto_seg_results = self.auto_segmenter.process_frame(image_bgr, prompt="Liquid", n_clusters=1)

                # Schedule UI update on main thread
                self.root.after(0, self.process_segmentation_result)

            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Liquid segmentation failed: {e}"))
            finally:
                self.is_processing = False
                self.root.after(0, self.update_button_states)  # Re-enable buttons when done
                self.root.after(0, lambda: self.update_status("Segmentation complete"))

        # Start the thread
        thread = threading.Thread(target=segmentation_worker, daemon=True)
        thread.start()

    def process_segmentation_result(self):
        """Process segmentation results on main thread"""
        try:
            if self.auto_seg_results and self.auto_seg_results['segmentation_mask'] is not None:
                self.segmentation_mask = self.auto_seg_results['segmentation_mask']
                # Store original segmentation for dual display
                self.original_segmentation_mask = self.segmentation_mask.copy()

                # Mark overlay for update
                self.overlay_needs_update = True

                pixel_count = np.sum(self.segmentation_mask)

                self.add_progress_log(f"Liquid detected: {pixel_count} pixels")

                # Step 1: Check capacity
                if pixel_count < MIN_CAPACITY_PIXELS:
                    self.add_progress_log("❌ CAPACITY NOT MATCH")
                    self.result_label.config(text="CAPACITY NOT MATCH")
                    self.usability_label.config(text="Insufficient liquid volume", background="#FFCDD2")
                    self.automation_state = AutomationState.ANALYSIS_COMPLETE
                    self.update_button_states()
                    return

                # Step 2: Check for layering with clustering (also heavy, run in thread)
                self.add_progress_log("Step 2: Checking for layering...")
                self.check_layering_threaded()

            else:
                messagebox.showwarning("Warning", "Liquid segmentation failed")

        except Exception as e:
            messagebox.showerror("Error", f"Processing segmentation failed: {e}")

    def check_layering_threaded(self):
        """Step 2: Check if liquid has layering using clustering (threaded version)"""
        # Capture current data for processing
        frame_to_process = self.current_frame.copy()
        mask_to_process = self.segmentation_mask.copy()

        def layering_worker():
            try:
                self.is_processing = True
                self.root.after(0, self.update_button_states)  # Disable buttons during processing
                self.root.after(0, lambda: self.update_status("Processing clustering analysis..."))

                # Do heavy clustering computation
                clustering_result = self._compute_clustering(frame_to_process, mask_to_process)

                # Schedule UI update on main thread
                self.root.after(0, lambda: self.process_layering_result(clustering_result))

            except Exception as e:
                self.root.after(0, lambda: self.add_progress_log(f"Layering check failed: {e}"))
                self.root.after(0, self.handle_single_liquid)
            finally:
                self.is_processing = False
                self.root.after(0, self.update_button_states)  # Re-enable buttons when done
                self.root.after(0, lambda: self.update_status("Clustering analysis complete"))

        # Start the thread
        thread = threading.Thread(target=layering_worker, daemon=True)
        thread.start()

    def _compute_clustering(self, image_rgb, segmentation_mask):
        """Compute clustering (called from worker thread)"""
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # Create middle region mask to avoid bottle edges
        h, w = segmentation_mask.shape
        middle_mask = np.zeros_like(segmentation_mask)
        start_x = w // 4
        end_x = 3 * w // 4
        middle_mask[:, start_x:end_x] = segmentation_mask[:, start_x:end_x]

        clustering = self.auto_segmenter.apply_kmeans_clustering(image_bgr, middle_mask, n_clusters=2)

        # Check if there's significant difference between clusters
        cluster_0_mask = clustering['cluster_masks'].get('cluster_0')
        cluster_1_mask = clustering['cluster_masks'].get('cluster_1')

        return {
            'cluster_0_mask': cluster_0_mask,
            'cluster_1_mask': cluster_1_mask,
            'image_rgb': image_rgb,
            'h': h,
            'w': w
        }

    def process_layering_result(self, clustering_result):
        """Process layering results on main thread"""
        try:
            cluster_0_mask = clustering_result['cluster_0_mask']
            cluster_1_mask = clustering_result['cluster_1_mask']
            roi_rgb = clustering_result['image_rgb']
            h = clustering_result['h']

            if cluster_0_mask is not None and cluster_1_mask is not None:
                area_0 = np.sum(cluster_0_mask)
                area_1 = np.sum(cluster_1_mask)

                # Check if both clusters have reasonable size
                total_area = area_0 + area_1
                ratio_0 = area_0 / total_area if total_area > 0 else 0
                ratio_1 = area_1 / total_area if total_area > 0 else 0

                # MUCH MORE STRICT criteria for layering detection
                min_cluster_ratio = 0.05

                # Check color difference between clusters
                mean_color_0 = cv2.mean(roi_rgb, mask=cluster_0_mask.astype(np.uint8))[:3]
                mean_color_1 = cv2.mean(roi_rgb, mask=cluster_1_mask.astype(np.uint8))[:3]

                # Calculate color distance (Euclidean distance in RGB space)
                color_diff = np.sqrt(np.sum([(mean_color_0[i] - mean_color_1[i])**2 for i in range(3)]))

                # Check vertical separation (clusters should be in different vertical regions)
                cluster_0_center_y = np.mean(np.where(cluster_0_mask)[0]) if np.any(cluster_0_mask) else 0
                cluster_1_center_y = np.mean(np.where(cluster_1_mask)[0]) if np.any(cluster_1_mask) else 0
                vertical_separation = abs(cluster_0_center_y - cluster_1_center_y)
                min_vertical_separation = h * 0.1  # At least 10% of image height separation

                self.add_progress_log(f"Cluster analysis:")
                self.add_progress_log(f"  - Ratio 0: {ratio_0:.3f}, Ratio 1: {ratio_1:.3f}")
                self.add_progress_log(f"  - Color difference: {color_diff:.2f}")
                self.add_progress_log(f"  - Vertical separation: {vertical_separation:.1f}px")

                # Very strict criteria: ALL conditions must be met
                layering_detected = (
                    ratio_0 >= min_cluster_ratio and
                    ratio_1 >= min_cluster_ratio and
                    color_diff > 25.0 and  # Significant color difference
                    vertical_separation > min_vertical_separation  # Vertical separation
                )

                if layering_detected:
                    self.add_progress_log(f"✓ STRONG layering detected (Areas: {area_0}, {area_1})")

                    # Store cluster data
                    bboxes = self.auto_segmenter.get_cluster_bboxes({'cluster_0': cluster_0_mask, 'cluster_1': cluster_1_mask})
                    self.current_clusters = {
                        'cluster_0': {
                            'mask': cluster_0_mask,
                            'bbox': bboxes.get('cluster_0', (0,0,0,0)),
                            'area': area_0
                        },
                        'cluster_1': {
                            'mask': cluster_1_mask,
                            'bbox': bboxes.get('cluster_1', (0,0,0,0)),
                            'area': area_1
                        }
                    }

                    # Mark overlay for update
                    self.overlay_needs_update = True

                    # Step 3: Wait for IR reading
                    self.automation_state = AutomationState.WAITING_IR_READ
                    self.update_button_states()
                    self.step_label.config(text="Step 3: Click IR READ button to read IR values")

                else:
                    self.add_progress_log("No significant layering found - treating as single liquid")
                    self.add_progress_log("  (Requires: >30% each cluster, >25 color diff, vertical separation)")
                    self.handle_single_liquid()

            else:
                self.add_progress_log("Clustering failed - treating as single liquid")
                self.handle_single_liquid()

        except Exception as e:
            self.add_progress_log(f"Layering processing failed: {e}")
            self.handle_single_liquid()

    def check_layering(self):
        """Step 2: Check if liquid has layering using clustering"""
        try:
            # Apply clustering to detect layering
            image_bgr = cv2.cvtColor(self.current_frame, cv2.COLOR_RGB2BGR)
            
            # Create middle region mask to avoid bottle edges
            h, w = self.segmentation_mask.shape
            middle_mask = np.zeros_like(self.segmentation_mask)
            start_x = w // 4
            end_x = 3 * w // 4
            middle_mask[:, start_x:end_x] = self.segmentation_mask[:, start_x:end_x]
            
            clustering = self.auto_segmenter.apply_kmeans_clustering(image_bgr, middle_mask, n_clusters=2)
            
            # Check if there's significant difference between clusters
            cluster_0_mask = clustering['cluster_masks'].get('cluster_0')
            cluster_1_mask = clustering['cluster_masks'].get('cluster_1')
            
            if cluster_0_mask is not None and cluster_1_mask is not None:
                area_0 = np.sum(cluster_0_mask)
                area_1 = np.sum(cluster_1_mask)
                
                # Check if both clusters have reasonable size
                total_area = area_0 + area_1
                ratio_0 = area_0 / total_area if total_area > 0 else 0
                ratio_1 = area_1 / total_area if total_area > 0 else 0
                
                # MUCH MORE STRICT criteria for layering detection
                # Both clusters must have at least 30% of the area (instead of 15%)
                min_cluster_ratio = 0.30
                
                # Check color difference between clusters
                roi_rgb = self.current_frame
                mean_color_0 = cv2.mean(roi_rgb, mask=cluster_0_mask.astype(np.uint8))[:3]
                mean_color_1 = cv2.mean(roi_rgb, mask=cluster_1_mask.astype(np.uint8))[:3]
                
                # Calculate color distance (Euclidean distance in RGB space)
                color_diff = np.sqrt(np.sum([(mean_color_0[i] - mean_color_1[i])**2 for i in range(3)]))
                
                # Check vertical separation (clusters should be in different vertical regions)
                cluster_0_center_y = np.mean(np.where(cluster_0_mask)[0]) if np.any(cluster_0_mask) else 0
                cluster_1_center_y = np.mean(np.where(cluster_1_mask)[0]) if np.any(cluster_1_mask) else 0
                vertical_separation = abs(cluster_0_center_y - cluster_1_center_y)
                min_vertical_separation = h * 0.1  # At least 10% of image height separation
                
                self.add_progress_log(f"Cluster analysis:")
                self.add_progress_log(f"  - Ratio 0: {ratio_0:.3f}, Ratio 1: {ratio_1:.3f}")
                self.add_progress_log(f"  - Color difference: {color_diff:.2f}")
                self.add_progress_log(f"  - Vertical separation: {vertical_separation:.1f}px")
                
                # Very strict criteria: ALL conditions must be met
                layering_detected = (
                    ratio_0 >= min_cluster_ratio and 
                    ratio_1 >= min_cluster_ratio and
                    color_diff > 25.0 and  # Significant color difference
                    vertical_separation > min_vertical_separation  # Vertical separation
                )
                
                if layering_detected:
                    self.add_progress_log(f"✓ STRONG layering detected (Areas: {area_0}, {area_1})")
                    
                    # Store cluster data
                    bboxes = self.auto_segmenter.get_cluster_bboxes(clustering['cluster_masks'])
                    self.current_clusters = {
                        'cluster_0': {
                            'mask': cluster_0_mask,
                            'bbox': bboxes.get('cluster_0', (0,0,0,0)),
                            'area': area_0
                        },
                        'cluster_1': {
                            'mask': cluster_1_mask,
                            'bbox': bboxes.get('cluster_1', (0,0,0,0)),
                            'area': area_1
                        }
                    }
                    
                    # Step 3: Wait for IR reading
                    self.automation_state = AutomationState.WAITING_IR_READ
                    self.update_button_states()
                    self.step_label.config(text="Step 3: Click IR READ button to read IR values")
                    
                else:
                    self.add_progress_log("No significant layering found - treating as single liquid")
                    self.add_progress_log("  (Requires: >30% each cluster, >25 color diff, vertical separation)")
                    self.handle_single_liquid()
                    
            else:
                self.add_progress_log("Clustering failed - treating as single liquid")
                self.handle_single_liquid()
                
        except Exception as e:
            self.add_progress_log(f"Layering check failed: {e}")
            self.handle_single_liquid()

    def handle_single_liquid(self):
        """Handle single liquid path (steps 6-10)"""
        # Step 6: Show middle 1/3 mask
        self.create_middle_third_mask()
        self.add_progress_log("Step 6: Using middle 1/3 segment")

        # Step 7: Wait for IR reading (even for single liquid)
        self.automation_state = AutomationState.WAITING_IR_READ
        self.update_button_states()
        self.step_label.config(text="Step 7: Click IR READ button to read IR value")

    def read_ir_values(self):
        """Step 4: Read IR values for each cluster OR single liquid"""
        self.add_progress_log("Step 4: Reading IR values...")
        
        try:
            if self.camera_available:
                frames = self.pipeline.wait_for_frames(timeout_ms=100)
                ir = np.asanyarray(frames.get_infrared_frame().get_data())
                if ROTATE_PREVIEW: ir = cv2.rotate(ir, ROTATE_DIR)
            else:
                ir = np.ones(self.current_frame.shape[:2], dtype=np.uint8) * 150

            # Check if we have clusters (dual liquid) or single liquid
            if self.current_clusters:
                # Dual liquid path - read IR for both clusters
                cluster_0_ir = cv2.mean(ir, mask=self.current_clusters['cluster_0']['mask'].astype(np.uint8))[0]
                cluster_1_ir = cv2.mean(ir, mask=self.current_clusters['cluster_1']['mask'].astype(np.uint8))[0]
                
                self.ir_values = {
                    'cluster_0': cluster_0_ir,
                    'cluster_1': cluster_1_ir
                }
                
                ir_diff = abs(cluster_0_ir - cluster_1_ir)
                
                self.add_progress_log(f"Cluster 0 IR: {cluster_0_ir:.2f}")
                self.add_progress_log(f"Cluster 1 IR: {cluster_1_ir:.2f}")
                self.add_progress_log(f"IR Difference: {ir_diff:.2f}")
                
                # Step 5: Check IR difference
                if ir_diff > 10:
                    self.add_progress_log("✓ Significant IR difference - analyzing both clusters")
                    self.handle_dual_liquid()
                else:
                    self.add_progress_log("No significant IR difference - treating as single liquid")
                    # Clear clusters and continue as single liquid
                    self.current_clusters = {}
                    self.automation_state = AutomationState.WAITING_RGB_SINGLE
                    self.update_button_states()
                    self.step_label.config(text="Step 8: Click RGB READ button to read RGB values")
            else:
                # Single liquid path - read IR for the segmented area
                single_ir = cv2.mean(ir, mask=self.segmentation_mask.astype(np.uint8))[0]

                self.ir_values = {'single': single_ir}

                self.add_progress_log(f"Single liquid IR: {single_ir:.2f}")

                # Move to RGB reading
                self.automation_state = AutomationState.WAITING_RGB_SINGLE
                self.update_button_states()
                self.step_label.config(text="Step 8: Click RGB READ button to read RGB values")
                
        except Exception as e:
            self.add_progress_log(f"IR reading failed: {e}")
            messagebox.showerror("Error", f"IR reading failed: {e}")

    def handle_dual_liquid(self):
        """Handle dual liquid path (steps 11-19)"""
        # Step 11: Show middle 1/3 mask for cluster 1
        self.create_cluster_middle_third_mask('cluster_0')
        self.add_progress_log("Step 11: Using middle 1/3 of cluster 0")

        # Step 12: Wait for RGB reading of cluster 1
        self.automation_state = AutomationState.WAITING_RGB_CLUSTER1
        self.update_button_states()
        self.step_label.config(text="Step 12: Click RGB READ button to read RGB for cluster 0")

    def read_rgb_single_liquid(self):
        """Step 8-10: Read RGB for single liquid and predict"""
        self.add_progress_log("Step 8: Reading RGB values...")
        
        try:
            # Extract features for current mask
            self.features = self.extract_features()
            if self.features is None:
                raise Exception("Feature extraction failed")

            # Calculate RGB scores
            roi_rgb = self.get_masked_roi()
            self.current_yellow_score = self.calculate_yellow_score(roi_rgb, self.segmentation_mask)
            self.current_black_score = self.calculate_black_score(roi_rgb, self.segmentation_mask)
            
            self.update_current_readings()
            
            # Step 9: Predict type
            self.add_progress_log("Step 9: Predicting liquid type...")
            
            # Use IR value from single liquid reading or extracted features
            if 'single' in self.ir_values:
                ir_val = self.ir_values['single']
            elif self.features is not None and len(self.features) > 11:
                ir_val = self.features[11]
            else:
                ir_val = 150  # fallback value
                
            result = self.classify_by_ir_and_rgb(ir_val, self.current_yellow_score, self.current_black_score)
            
            pixel_count = np.sum(self.segmentation_mask)
            
            # Step 10: Show results
            self.add_progress_log("Step 10: Analysis complete")
            self.add_progress_log(f"Type: {result['class']}")
            self.add_progress_log(f"Confidence: {result['confidence']:.3f}")
            self.add_progress_log(f"Pixels: {pixel_count}")
            
            self.show_single_liquid_result(result, pixel_count)
            
        except Exception as e:
            self.add_progress_log(f"RGB reading failed: {e}")
            messagebox.showerror("Error", f"RGB reading failed: {e}")

    def read_rgb_cluster1(self):
        """Step 13: Read RGB for cluster 0"""
        self.add_progress_log("Step 13: Reading RGB for cluster 0...")
        
        try:
            # Use cluster 0 mask
            mask = self.current_clusters['cluster_0']['mask']
            roi_rgb = self.get_masked_roi()
            
            yellow_score = self.calculate_yellow_score(roi_rgb, mask)
            black_score = self.calculate_black_score(roi_rgb, mask)
            
            self.rgb_values['cluster_0'] = {
                'yellow': yellow_score,
                'black': black_score
            }
            
            self.add_progress_log(f"Cluster 0 - Yellow: {yellow_score:.2f}, Black: {black_score:.2f}")

            # Step 14: Move to cluster 1
            self.create_cluster_middle_third_mask('cluster_1')
            self.add_progress_log("Step 14: Using middle 1/3 of cluster 1")

            self.automation_state = AutomationState.WAITING_RGB_CLUSTER2
            self.update_button_states()
            self.step_label.config(text="Step 15: Click RGB READ button to read RGB for cluster 1")
            
        except Exception as e:
            self.add_progress_log(f"RGB cluster 0 reading failed: {e}")
            messagebox.showerror("Error", f"RGB cluster 0 reading failed: {e}")

    def read_rgb_cluster2(self):
        """Step 16: Read RGB for cluster 1 and complete analysis"""
        self.add_progress_log("Step 16: Reading RGB for cluster 1...")
        
        try:
            # Use cluster 1 mask
            mask = self.current_clusters['cluster_1']['mask']
            roi_rgb = self.get_masked_roi()
            
            yellow_score = self.calculate_yellow_score(roi_rgb, mask)
            black_score = self.calculate_black_score(roi_rgb, mask)
            
            self.rgb_values['cluster_1'] = {
                'yellow': yellow_score,
                'black': black_score
            }
            
            self.add_progress_log(f"Cluster 1 - Yellow: {yellow_score:.2f}, Black: {black_score:.2f}")
            
            # Step 17: Predict both types
            self.add_progress_log("Step 17: Predicting both liquid types...")
            self.analyze_dual_liquids()
            
        except Exception as e:
            self.add_progress_log(f"RGB cluster 1 reading failed: {e}")
            messagebox.showerror("Error", f"RGB cluster 1 reading failed: {e}")

    def analyze_dual_liquids(self):
        """Steps 17-19: Analyze dual liquids and determine usability"""
        try:
            # Predict types for both clusters
            results = {}
            for cluster_name in ['cluster_0', 'cluster_1']:
                ir_val = self.ir_values[cluster_name]
                yellow = self.rgb_values[cluster_name]['yellow']
                black = self.rgb_values[cluster_name]['black']
                
                result = self.classify_by_ir_and_rgb(ir_val, yellow, black)
                results[cluster_name] = result
                
                self.add_progress_log(f"{cluster_name}: {result['class']} (conf: {result['confidence']:.3f})")
            
            # Step 18: Check if both clusters are usable
            cluster_0_type = results['cluster_0']['class']
            cluster_1_type = results['cluster_1']['class']

            usable_oils = ['Light oil', 'Medium oil', 'Dark oil']
            cluster_0_usable = cluster_0_type in usable_oils
            cluster_1_usable = cluster_1_type in usable_oils

            # Check if both clusters are usable oil types
            if cluster_0_usable and cluster_1_usable:
                self.add_progress_log("Step 18: Both clusters are usable oil types")
                self.show_dual_liquid_result(results, usable=True, reason="Both clusters are usable oils")
                return

            # Step 19: Check water exception - one cluster is water with <50%
            water_cluster = None
            oil_cluster = None

            if cluster_0_type == 'Water' or 'Water' in cluster_0_type:
                water_cluster = 'cluster_0'
                oil_cluster = 'cluster_1'
            elif cluster_1_type == 'Water' or 'Water' in cluster_1_type:
                water_cluster = 'cluster_1'
                oil_cluster = 'cluster_0'

            # If one cluster is water, check the water percentage exception
            if water_cluster and oil_cluster:
                oil_type = results[oil_cluster]['class']
                oil_usable = oil_type in usable_oils

                if oil_usable:
                    water_area = self.current_clusters[water_cluster]['area']
                    oil_area = self.current_clusters[oil_cluster]['area']
                    total_area = water_area + oil_area
                    water_percentage = (water_area / total_area) * 100 if total_area > 0 else 0

                    self.add_progress_log(f"Water percentage: {water_percentage:.1f}%")

                    if water_percentage < 50:
                        self.add_progress_log("Step 19: Water level acceptable (<50%)")
                        self.show_dual_liquid_result(results, usable=True, reason=f"Water <50% ({water_percentage:.1f}%) with usable oil")
                    else:
                        self.add_progress_log("Step 19: Too much water (>=50%)")
                        self.show_dual_liquid_result(results, usable=False, reason=f"Too much water ({water_percentage:.1f}%)")
                else:
                    self.add_progress_log("Step 18: Oil cluster is not a usable type")
                    self.show_dual_liquid_result(results, usable=False, reason="Non-usable oil type with water")
            else:
                # No water, but not both usable oils - mark as unusable
                self.add_progress_log("Step 18: Both clusters must be usable oil types")
                self.show_dual_liquid_result(results, usable=False, reason="Both clusters are not usable oil types")
                
        except Exception as e:
            self.add_progress_log(f"Dual liquid analysis failed: {e}")
            messagebox.showerror("Error", f"Dual liquid analysis failed: {e}")

    def create_middle_third_mask(self):
        """Create a mask for the middle 1/3 of the current segmentation"""
        # Use original segmentation mask if available, otherwise use current one
        if hasattr(self, 'original_segmentation_mask') and self.original_segmentation_mask is not None:
            source_mask = self.original_segmentation_mask
        elif self.segmentation_mask is not None:
            source_mask = self.segmentation_mask
        else:
            return

        # Get bounding box of source mask
        rows = np.any(source_mask, axis=1)
        cols = np.any(source_mask, axis=0)

        if not np.any(rows) or not np.any(cols):
            return

        y_min, y_max = np.where(rows)[0][[0, -1]]
        height = y_max - y_min

        # Calculate middle 1/3 vertical range
        new_y_min = int(y_min + height / 3)
        new_y_max = int(y_min + 2 * height / 3)

        # Create new mask from source
        new_mask = np.zeros_like(source_mask)
        new_mask[new_y_min:new_y_max, :] = source_mask[new_y_min:new_y_max, :]

        self.segmentation_mask = new_mask

        # Update bbox
        rows_new = np.any(new_mask, axis=1)
        cols_new = np.any(new_mask, axis=0)
        if np.any(rows_new) and np.any(cols_new):
            ny_min, ny_max = np.where(rows_new)[0][[0, -1]]
            nx_min, nx_max = np.where(cols_new)[0][[0, -1]]
            self.bbox_coords = (nx_min, ny_min, nx_max, ny_max)

        # Mark overlay for update
        self.overlay_needs_update = True

    def create_cluster_middle_third_mask(self, cluster_name):
        """Create a middle 1/3 mask for a specific cluster"""
        if cluster_name not in self.current_clusters:
            return

        cluster_mask = self.current_clusters[cluster_name]['mask']
        if cluster_mask is None:
            return

        # Get bounding box of cluster mask
        rows = np.any(cluster_mask, axis=1)
        cols = np.any(cluster_mask, axis=0)

        if not np.any(rows) or not np.any(cols):
            return

        y_min, y_max = np.where(rows)[0][[0, -1]]
        height = y_max - y_min

        # Calculate middle 1/3 vertical range
        new_y_min = int(y_min + height / 3)
        new_y_max = int(y_min + 2 * height / 3)

        # Create new mask
        new_mask = np.zeros_like(cluster_mask)
        new_mask[new_y_min:new_y_max, :] = cluster_mask[new_y_min:new_y_max, :]

        # Update current segmentation mask to this cluster's middle third
        self.segmentation_mask = new_mask

        # Update bbox
        rows_new = np.any(new_mask, axis=1)
        cols_new = np.any(new_mask, axis=0)
        if np.any(rows_new) and np.any(cols_new):
            ny_min, ny_max = np.where(rows_new)[0][[0, -1]]
            nx_min, nx_max = np.where(cols_new)[0][[0, -1]]
            self.bbox_coords = (nx_min, ny_min, nx_max, ny_max)

        # Mark overlay for update
        self.overlay_needs_update = True

    def get_masked_roi(self):
        """Get the masked ROI for RGB analysis"""
        if self.current_frame is None or self.segmentation_mask is None:
            return None
        return self.current_frame

    def extract_features(self):
        """Extract features for classification"""
        if self.current_frame is None or self.segmentation_mask is None: 
            return None
        try:
            if self.camera_available:
                frames = self.pipeline.wait_for_frames(timeout_ms=100)
                ir = np.asanyarray(frames.get_infrared_frame().get_data())
                if ROTATE_PREVIEW: ir = cv2.rotate(ir, ROTATE_DIR)
            else: 
                ir = np.ones(self.current_frame.shape[:2], dtype=np.uint8) * 150

            # Get bounding box
            rows = np.any(self.segmentation_mask, axis=1)
            cols = np.any(self.segmentation_mask, axis=0)
            if not np.any(rows) or not np.any(cols):
                return None
            
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]
            self.bbox_coords = (x_min, y_min, x_max, y_max)

            x1, y1, x2, y2 = self.bbox_coords
            roi_col = self.current_frame[y1:y2, x1:x2]
            roi_ir = ir[y1:y2, x1:x2]
            
            mask = np.zeros(self.current_frame.shape[:2], dtype=np.uint8)
            mask[y1:y2, x1:x2] = 255
            mask_roi = mask[y1:y2, x1:x2]
            roi_bgr = cv2.cvtColor(roi_col, cv2.COLOR_RGB2BGR)

            # Standard features
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

    def show_dual_liquid_result(self, results, usable, reason):
        """Display results for dual liquid analysis"""
        cluster_0_result = results['cluster_0']
        cluster_1_result = results['cluster_1']
        
        result_text = f"Cluster 0: {cluster_0_result['class']}\nCluster 1: {cluster_1_result['class']}"
        self.result_label.config(text=result_text)
        
        bg_color = "#C8E6C9" if usable else "#FFCDD2"
        status_text = "USABLE" if usable else "UNUSABLE"
        self.usability_label.config(text=f"{status_text} - {reason}", background=bg_color)
        
        # Display distance analysis for both clusters
        self.distances_text.delete(1.0, tk.END)
        self.distances_text.insert(tk.END, "=== DUAL LIQUID DISTANCES ===\n")
        
        for cluster_name, result in results.items():
            self.distances_text.insert(tk.END, f"\n{cluster_name.upper()}:\n")
            self.distances_text.insert(tk.END, f"  Type: {result['class']}\n")
            self.distances_text.insert(tk.END, f"  Confidence: {result['confidence']:.3f}\n")
            self.distances_text.insert(tk.END, f"  Distance: {result['total_distance']:.4f}\n")
        
        self.details_text.delete(1.0, tk.END)
        self.details_text.insert(tk.END, "=== ENHANCED DUAL LIQUID ANALYSIS ===\n\n")
        
        for cluster_name, result in results.items():
            area = self.current_clusters[cluster_name]['area']
            self.details_text.insert(tk.END, f"{cluster_name.upper()}:\n")
            self.details_text.insert(tk.END, f"  Type: {result['class']}\n")
            self.details_text.insert(tk.END, f"  Confidence: {result['confidence']:.3f}\n")
            self.details_text.insert(tk.END, f"  Pixels: {area}\n")
            self.details_text.insert(tk.END, f"  IR: {result['ir_value']:.2f}\n")
            self.details_text.insert(tk.END, f"  Yellow: {result['yellow_score']:.2f}\n")
            self.details_text.insert(tk.END, f"  Black: {result['black_score']:.2f}\n")
            
            # Show thresholds and distances
            thresh = result['thresholds']
            comp_dist = result['component_distances']
            self.details_text.insert(tk.END, f"  Thresholds: IR={thresh['ir']:.1f}, Y={thresh['yellow']:.1f}, B={thresh['black']:.1f}\n")
            self.details_text.insert(tk.END, f"  Distances: IR={comp_dist['ir']:.2f}, Y={comp_dist['yellow']:.2f}, B={comp_dist['black']:.2f}\n\n")
        
        self.details_text.insert(tk.END, f"Final Status: {status_text}\n")
        self.details_text.insert(tk.END, f"Reason: {reason}\n")
        
        self.automation_state = AutomationState.ANALYSIS_COMPLETE
        self.update_button_states()
        self.step_label.config(text="Analysis Complete")

def main():
    root = tk.Tk()
    app = OilClassificationEnhancedGUI(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.cleanup(), root.destroy()))
    root.mainloop()

if __name__ == "__main__":
    main()