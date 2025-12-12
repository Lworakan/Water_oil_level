import numpy as np
import cv2
from typing import Tuple, List, Dict, Optional


class ColorFeatureExtractor:
    """Extract color-based features from images for oil quality analysis"""
    
    def __init__(self):
        self.feature_names = [
            'color_bbox_mean_r', 'color_bbox_mean_g', 'color_bbox_mean_b',
            'color_bbox_std_r', 'color_bbox_std_g', 'color_bbox_std_b',
            'color_poly_mean_r', 'color_poly_mean_g', 'color_poly_mean_b',
            'color_poly_std_r', 'color_poly_std_g', 'color_poly_std_b',
            'hsv_bbox_mean_h', 'hsv_bbox_mean_s', 'hsv_bbox_mean_v',
            'hsv_poly_mean_h', 'hsv_poly_mean_s', 'hsv_poly_mean_v',
            'color_bbox_entropy', 'color_poly_entropy',
            'color_bbox_contrast', 'color_poly_contrast'
        ]
    
    def extract_realtime_features(self, img_color: np.ndarray, 
                                 bbox_coords: Tuple[int, int, int, int],
                                 poly_points: List[List[int]] = None) -> Dict[str, float]:
        """
        Extract color features in real-time for GUI integration
        
        Args:
            img_color: Color image (BGR format)
            bbox_coords: Bounding box coordinates (x1, y1, x2, y2)
            poly_points: List of polygon points [[x, y], ...] (optional)
        
        Returns:
            Dictionary of color features
        """
        features = {}
        
        # Extract bbox features
        bbox_features = self._extract_bbox_features(img_color, bbox_coords)
        features.update(bbox_features)
        
        # Extract poly features if points provided
        if poly_points and len(poly_points) >= 3:
            poly_features = self._extract_poly_features(img_color, poly_points)
            features.update(poly_features)
        else:
            # Add empty poly features
            for name in self.feature_names:
                if 'poly' in name:
                    features[name] = 0.0
        
        return features
    
    def _extract_bbox_features(self, img: np.ndarray, bbox: Tuple[int, int, int, int]) -> Dict[str, float]:
        """Extract color features from bounding box region"""
        x1, y1, x2, y2 = bbox
        
        # Ensure valid coordinates
        h, w = img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return self._get_empty_bbox_features()
        
        roi = img[y1:y2, x1:x2]
        
        features = {}
        
        # RGB statistics
        for i, channel in enumerate(['r', 'g', 'b']):
            channel_data = roi[:, :, 2-i]  # BGR to RGB conversion
            features[f'color_bbox_mean_{channel}'] = float(np.mean(channel_data))
            features[f'color_bbox_std_{channel}'] = float(np.std(channel_data))
        
        # HSV statistics
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        features['hsv_bbox_mean_h'] = float(np.mean(hsv_roi[:, :, 0]))
        features['hsv_bbox_mean_s'] = float(np.mean(hsv_roi[:, :, 1]))
        features['hsv_bbox_mean_v'] = float(np.mean(hsv_roi[:, :, 2]))
        
        # Texture features
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        features['color_bbox_entropy'] = float(self._calculate_entropy(gray_roi))
        features['color_bbox_contrast'] = float(self._calculate_contrast(gray_roi))
        
        return features
    
    def _extract_poly_features(self, img: np.ndarray, poly_points: List[List[int]]) -> Dict[str, float]:
        """Extract color features from polygon region"""
        if len(poly_points) < 3:
            return self._get_empty_poly_features()
        
        # Create mask for polygon
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        poly_array = np.array(poly_points, dtype=np.int32)
        cv2.fillPoly(mask, [poly_array], 255)
        
        # Extract pixels within polygon
        if np.sum(mask) == 0:
            return self._get_empty_poly_features()
        
        features = {}
        
        # RGB statistics
        for i, channel in enumerate(['r', 'g', 'b']):
            channel_data = img[:, :, 2-i]  # BGR to RGB conversion
            masked_pixels = channel_data[mask == 255]
            if len(masked_pixels) > 0:
                features[f'color_poly_mean_{channel}'] = float(np.mean(masked_pixels))
                features[f'color_poly_std_{channel}'] = float(np.std(masked_pixels))
            else:
                features[f'color_poly_mean_{channel}'] = 0.0
                features[f'color_poly_std_{channel}'] = 0.0
        
        # HSV statistics
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        for i, channel in enumerate(['h', 's', 'v']):
            channel_data = hsv_img[:, :, i]
            masked_pixels = channel_data[mask == 255]
            if len(masked_pixels) > 0:
                features[f'hsv_poly_mean_{channel}'] = float(np.mean(masked_pixels))
            else:
                features[f'hsv_poly_mean_{channel}'] = 0.0
        
        # Texture features
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_masked = gray[mask == 255]
        if len(gray_masked) > 0:
            features['color_poly_entropy'] = float(self._calculate_entropy_1d(gray_masked))
            features['color_poly_contrast'] = float(self._calculate_contrast_1d(gray_masked))
        else:
            features['color_poly_entropy'] = 0.0
            features['color_poly_contrast'] = 0.0
        
        return features
    
    def _calculate_entropy(self, img: np.ndarray) -> float:
        """Calculate Shannon entropy of image"""
        hist, _ = np.histogram(img.ravel(), bins=256, range=(0, 256))
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        if len(hist) == 0:
            return 0.0
        return -np.sum(hist * np.log2(hist))
    
    def _calculate_entropy_1d(self, pixels: np.ndarray) -> float:
        """Calculate Shannon entropy for 1D pixel array"""
        if len(pixels) == 0:
            return 0.0
        hist, _ = np.histogram(pixels, bins=256, range=(0, 256))
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        if len(hist) == 0:
            return 0.0
        return -np.sum(hist * np.log2(hist))
    
    def _calculate_contrast(self, img: np.ndarray) -> float:
        """Calculate contrast using standard deviation"""
        return np.std(img)
    
    def _calculate_contrast_1d(self, pixels: np.ndarray) -> float:
        """Calculate contrast for 1D pixel array"""
        if len(pixels) == 0:
            return 0.0
        return np.std(pixels)
    
    def _get_empty_bbox_features(self) -> Dict[str, float]:
        """Return empty bbox features dictionary"""
        features = {}
        for name in self.feature_names:
            if 'bbox' in name:
                features[name] = 0.0
        return features
    
    def _get_empty_poly_features(self) -> Dict[str, float]:
        """Return empty poly features dictionary"""
        features = {}
        for name in self.feature_names:
            if 'poly' in name:
                features[name] = 0.0
        return features
    
    def get_csv_headers(self) -> List[str]:
        """Get list of feature names for CSV headers"""
        return self.feature_names.copy()
    
    def features_to_csv_row(self, features: Dict[str, float]) -> List[str]:
        """Convert features dictionary to CSV row values"""
        return [f"{features.get(name, 0.0):.3f}" for name in self.feature_names]
    
    def extract_bbox_features(self, img: np.ndarray, bbox: Tuple[int, int, int, int]) -> Dict[str, float]:
        """Extract color features from bounding box region (public method)"""
        return self._extract_bbox_features(img, bbox)
    
    def extract_poly_features(self, img: np.ndarray, poly_points: List[List[int]]) -> Dict[str, float]:
        """Extract color features from polygon region (public method)"""
        return self._extract_poly_features(img, poly_points)