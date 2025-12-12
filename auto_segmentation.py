import os
import cv2
import torch
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional, Any
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")

try:
    from transformers import Sam3Processor, Sam3Model
    SAM_AVAILABLE = True
except ImportError:
    print("Warning: SAM3 not available. Install transformers: pip install transformers")
    SAM_AVAILABLE = False


class AutoSegmenter:
    """Auto segmentation using SAM + K-means clustering for oil quality analysis"""
    
    def __init__(self, model_name: str = "facebook/sam3", device: Optional[str] = None, use_sam: bool = True):
        """
        Initialize the auto segmenter
        
        Args:
            model_name: SAM model to use
            device: Device for model inference
            use_sam: Whether to attempt loading SAM model (default False for stability)
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self.sam_available = SAM_AVAILABLE and use_sam
        
        if self.sam_available:
            try:
                print(f"Loading SAM3 model: {model_name} on {self.device}")
                self.model = Sam3Model.from_pretrained(model_name).to(self.device)
                self.processor = Sam3Processor.from_pretrained(model_name)
                print("SAM3 model loaded successfully")
            except Exception as e:
                print(f"Warning: Could not load SAM3 model: {e}")
                self.sam_available = False
        
        # Use traditional CV methods by default for reliability
        if not self.sam_available:
            print("Using traditional computer vision methods for segmentation")
    
    def segment_bottle_region(self, image: np.ndarray, prompt: str = "bottle") -> Optional[np.ndarray]:
        """
        Segment bottle region from image
        
        Args:
            image: Input BGR image
            prompt: Text prompt for segmentation
            
        Returns:
            Binary mask of bottle region or None if failed
        """
        if self.sam_available and self.model is not None:
            return self._sam_segmentation(image, prompt)
        else:
            return self._fallback_segmentation(image)
    
    def _sam_segmentation(self, image: np.ndarray, prompt: str) -> Optional[np.ndarray]:
        """SAM3-based segmentation using text prompts"""
        try:
            # Convert BGR to RGB for SAM3
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            print(f"Segmenting with SAM3 prompt: '{prompt}'...")
            
            # Process with SAM3 (supports text prompts)
            inputs = self.processor(images=pil_image, text=prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Post-process results
            results = self.processor.post_process_instance_segmentation(
                outputs,
                threshold=0.5,
                mask_threshold=0.5,
                target_sizes=inputs.get("original_sizes").tolist()
            )[0]
            
            print(f"SAM3 found {len(results['masks'])} objects.")
            
            if len(results['masks']) > 0:
                # Combine all masks
                combined_mask = torch.any(results['masks'], dim=0).cpu().numpy().astype(np.uint8)
                return combined_mask
            
        except Exception as e:
            print(f"SAM3 segmentation failed: {e}")
            return None
        
        return None
    
    def _fallback_segmentation(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Fallback segmentation using traditional CV methods"""
        try:
            # Method 1: Edge-based detection for bottle shape
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Edge detection
            edges = cv2.Canny(blurred, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Filter contours by area and aspect ratio (bottle-like shapes)
                valid_contours = []
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 1000:  # Minimum area threshold
                        x, y, w, h = cv2.boundingRect(contour)
                        aspect_ratio = h / w if w > 0 else 0
                        if 1.5 < aspect_ratio < 4.0:  # Bottle-like aspect ratio
                            valid_contours.append(contour)
                
                if valid_contours:
                    # Use largest valid contour
                    largest_contour = max(valid_contours, key=cv2.contourArea)
                    mask = np.zeros(gray.shape, dtype=np.uint8)
                    cv2.fillPoly(mask, [largest_contour], 255)
                    return (mask / 255).astype(np.uint8)
            
            # Method 2: HSV-based segmentation as backup
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Create mask for non-background regions
            # Adjust these bounds based on your setup background
            lower_bound = np.array([0, 10, 50])     # Lower HSV bound
            upper_bound = np.array([180, 255, 255]) # Upper HSV bound
            
            mask = cv2.inRange(hsv, lower_bound, upper_bound)
            
            # Clean up the mask
            kernel = np.ones((7, 7), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Find largest contour
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest_contour) > 5000:  # Minimum size
                    mask = np.zeros_like(mask)
                    cv2.fillPoly(mask, [largest_contour], 255)
                    return (mask / 255).astype(np.uint8)
            
        except Exception as e:
            print(f"Fallback segmentation failed: {e}")
        
        return None
    
    def apply_kmeans_clustering(self, image: np.ndarray, mask: np.ndarray, 
                               n_clusters: int = 2) -> Dict[str, Any]:
        """
        Apply K-means clustering to segmented region
        
        Args:
            image: Original BGR image
            mask: Binary mask of segmented region
            n_clusters: Number of clusters
            
        Returns:
            Dictionary containing clustering results
        """
        if mask is None or np.sum(mask) == 0:
            return self._empty_clustering_result(n_clusters)
        
        # Extract pixels in masked region
        masked_pixels = image[mask == 1]
        
        if len(masked_pixels) == 0:
            return self._empty_clustering_result(n_clusters)
        
        # Apply K-means clustering
        pixels_reshaped = masked_pixels.reshape(-1, 3)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(pixels_reshaped)
        
        # Calculate statistics
        unique, counts = np.unique(labels, return_counts=True)
        total_pixels = len(labels)
        
        cluster_stats = {}
        cluster_masks = {}
        
        # Create cluster masks and calculate statistics
        full_labels = np.full(image.shape[:2], -1, dtype=int)
        full_labels[mask == 1] = labels
        
        for i, (cluster_id, count) in enumerate(zip(unique, counts)):
            percentage = (count / total_pixels) * 100
            center_color = kmeans.cluster_centers_[cluster_id]
            
            # Create individual cluster mask
            cluster_mask = (full_labels == cluster_id).astype(np.uint8)
            cluster_masks[f'cluster_{cluster_id}'] = cluster_mask
            
            cluster_stats[f'cluster_{cluster_id}'] = {
                'percentage': percentage,
                'pixel_count': count,
                'center_color_bgr': center_color.astype(int).tolist(),
                'center_color_rgb': center_color[::-1].astype(int).tolist()  # BGR to RGB
            }
        
        return {
            'cluster_stats': cluster_stats,
            'cluster_masks': cluster_masks,
            'combined_mask': mask,
            'full_labels': full_labels,
            'kmeans_centers': kmeans.cluster_centers_,
            'n_clusters': len(unique)
        }
    
    def get_cluster_bboxes(self, cluster_masks: Dict[str, np.ndarray]) -> Dict[str, Tuple[int, int, int, int]]:
        """
        Get bounding boxes for each cluster
        
        Args:
            cluster_masks: Dictionary of cluster masks
            
        Returns:
            Dictionary of cluster bounding boxes (x1, y1, x2, y2)
        """
        bboxes = {}
        
        for cluster_name, mask in cluster_masks.items():
            if np.sum(mask) > 0:
                coords = np.where(mask == 1)
                y_min, y_max = np.min(coords[0]), np.max(coords[0])
                x_min, x_max = np.min(coords[1]), np.max(coords[1])
                bboxes[cluster_name] = (x_min, y_min, x_max, y_max)
            else:
                bboxes[cluster_name] = (0, 0, 0, 0)
        
        return bboxes
    
    def get_cluster_polygons(self, cluster_masks: Dict[str, np.ndarray]) -> Dict[str, List[List[int]]]:
        """
        Get polygon contours for each cluster
        
        Args:
            cluster_masks: Dictionary of cluster masks
            
        Returns:
            Dictionary of cluster polygon points
        """
        polygons = {}
        
        for cluster_name, mask in cluster_masks.items():
            if np.sum(mask) > 0:
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    # Get largest contour
                    largest_contour = max(contours, key=cv2.contourArea)
                    # Simplify contour
                    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
                    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                    polygons[cluster_name] = approx.reshape(-1, 2).tolist()
                else:
                    polygons[cluster_name] = []
            else:
                polygons[cluster_name] = []
        
        return polygons
    
    def visualize_clustering(self, image: np.ndarray, clustering_result: Dict[str, Any]) -> np.ndarray:
        """
        Create visualization of clustering results
        
        Args:
            image: Original image
            clustering_result: Results from apply_kmeans_clustering
            
        Returns:
            Visualization image
        """
        if clustering_result['n_clusters'] == 0:
            return image.copy()
        
        result_img = image.copy()
        overlay = result_img.copy()
        
        # Visualization colors
        vis_colors = [
            [0, 255, 255],   # Yellow
            [255, 0, 255],   # Magenta  
            [255, 255, 0],   # Cyan
            [0, 255, 0],     # Green
            [0, 0, 255],     # Red
            [255, 0, 0],     # Blue
        ]
        
        # Apply cluster colors
        full_labels = clustering_result['full_labels']
        for i, (cluster_name, stats) in enumerate(clustering_result['cluster_stats'].items()):
            cluster_id = int(cluster_name.split('_')[1])
            color = vis_colors[i % len(vis_colors)]
            overlay[full_labels == cluster_id] = color
        
        # Blend with original
        alpha = 0.4
        result_img = cv2.addWeighted(result_img, 1 - alpha, overlay, alpha, 0)
        
        # Add cluster boundaries
        for cluster_name, mask in clustering_result['cluster_masks'].items():
            if np.sum(mask) > 0:
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(result_img, contours, -1, (255, 255, 255), 2)
        
        # Add text statistics
        y_offset = 30
        for cluster_name, stats in clustering_result['cluster_stats'].items():
            text = f"{cluster_name}: {stats['percentage']:.1f}%"
            cv2.putText(result_img, text, (10, y_offset), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25
        
        return result_img
    
    def _empty_clustering_result(self, n_clusters: int) -> Dict[str, Any]:
        """Return empty clustering result"""
        return {
            'cluster_stats': {},
            'cluster_masks': {},
            'combined_mask': np.zeros((1, 1), dtype=np.uint8),
            'full_labels': np.full((1, 1), -1, dtype=int),
            'kmeans_centers': np.zeros((n_clusters, 3)),
            'n_clusters': 0
        }
    
    def process_frame(self, image: np.ndarray, prompt: str = "bottle", 
                     n_clusters: int = 2) -> Dict[str, Any]:
        """
        Complete processing pipeline for a single frame
        
        Args:
            image: Input BGR image
            prompt: Segmentation prompt
            n_clusters: Number of clusters for K-means
            
        Returns:
            Complete analysis results
        """
        # Step 1: Segment bottle region
        mask = self.segment_bottle_region(image, prompt)
        
        # Step 2: Apply clustering
        clustering_result = self.apply_kmeans_clustering(image, mask, n_clusters)
        
        # Step 3: Get geometric features
        bboxes = self.get_cluster_bboxes(clustering_result['cluster_masks'])
        polygons = self.get_cluster_polygons(clustering_result['cluster_masks'])
        
        # Step 4: Create visualization
        visualization = self.visualize_clustering(image, clustering_result)
        
        return {
            'segmentation_mask': mask,
            'clustering': clustering_result,
            'bboxes': bboxes,
            'polygons': polygons,
            'visualization': visualization
        }


def test_segmentation(image_path: str):
    """Test function to verify segmentation works"""
    if not os.path.exists(image_path):
        print(f"Test image not found: {image_path}")
        return
    
    # Load test image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load image: {image_path}")
        return
    
    print(f"Testing segmentation on: {image_path}")
    
    # Create segmenter (try SAM3 first, fallback to CV if needed)
    segmenter = AutoSegmenter(use_sam=True)
    
    # Process the image
    result = segmenter.process_frame(img, n_clusters=2)
    
    if result['segmentation_mask'] is not None:
        print(f"✓ Segmentation successful")
        print(f"✓ Found {result['clustering']['n_clusters']} clusters")
        
        # Save visualization if successful
        output_path = "test_segmentation_result.jpg"
        cv2.imwrite(output_path, result['visualization'])
        print(f"✓ Visualization saved to: {output_path}")
        
        # Print cluster statistics
        for cluster_name, stats in result['clustering']['cluster_stats'].items():
            print(f"  {cluster_name}: {stats['percentage']:.1f}% pixels")
    else:
        print("✗ Segmentation failed")


if __name__ == "__main__":
    # Test with a sample image if available
    test_image = "test_bottle.jpg"  # Replace with actual test image path
    test_segmentation(test_image)