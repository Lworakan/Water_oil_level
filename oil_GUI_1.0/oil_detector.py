"""
Oil detection and volume calculation logic
"""

import cv2
import numpy as np
from config import *


class OilDetector:
    """Handles oil detection, volume calculation, and classification"""

    def __init__(self, bottles):
        """
        Initialize detector

        Args:
            bottles: List of OilBottle objects from reference chart
        """
        self.bottles = bottles
        self.gradient_bar = None
        if bottles:
            self.create_gradient_bar()

    def analyze_sample(self, frame, roi=None):
        """
        Analyze oil sample from camera frame

        Args:
            frame: Camera frame (BGR)
            roi: Region of interest (x1, y1, x2, y2), or None for center 50%

        Returns:
            Dictionary with color data
        """
        if roi is None:
            h, w = frame.shape[:2]
            x1, y1 = w // 4, h // 4
            x2, y2 = 3 * w // 4, 3 * h // 4
            roi = (x1, y1, x2, y2)

        x1, y1, x2, y2 = roi
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

    def detect_oil_volume(self, frame, roi=None):
        """
        Detect oil volume by separating oil from empty space using binary preprocessing

        Logic:
        - White/bright pixels (brightness >= WHITE_THRESHOLD) = empty space/water
        - Colored pixels (yellow, orange, brown, dark) = oil

        Args:
            frame: Camera frame (BGR)
            roi: Region of interest, or None for center 50%

        Returns:
            Tuple of (volume_liters, fill_percent, oil_pixel_count, total_pixels, binary_mask)
        """
        if roi is None:
            h, w = frame.shape[:2]
            x1, y1 = w // 4, h // 4
            x2, y2 = 3 * w // 4, 3 * h // 4
            roi = (x1, y1, x2, y2)

        x1, y1, x2, y2 = roi
        sample = frame[y1:y2, x1:x2]

        # Convert to grayscale for preprocessing
        gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)

        # Create binary mask using threshold
        # Pixels BELOW threshold (darker) = 255 (white in binary = oil)
        # Pixels ABOVE threshold (brighter) = 0 (black in binary = empty)
        _, binary_mask = cv2.threshold(gray, WHITE_THRESHOLD, 255, cv2.THRESH_BINARY_INV)

        # Apply morphological operations to clean up noise
        kernel = np.ones((3, 3), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        # Count oil pixels (white pixels in binary mask)
        oil_pixels = np.sum(binary_mask == 255)
        total_pixels = binary_mask.size

        # Calculate fill percentage
        if total_pixels > 0:
            fill_percent = (oil_pixels / total_pixels) * 100.0
        else:
            fill_percent = 0.0

        # Clamp between 0-100%
        fill_percent = max(0.0, min(100.0, fill_percent))

        # Calculate volume in liters
        volume_liters = (fill_percent / 100.0) * CONTAINER_CAPACITY_LITERS

        # Convert binary mask to 3-channel for display
        binary_display = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)

        return volume_liters, fill_percent, oil_pixels, total_pixels, binary_display

    def is_water(self, color_data):
        """
        Check if the sample is water/no oil

        Args:
            color_data: Color data dictionary

        Returns:
            True if water/no oil detected
        """
        brightness = color_data['hsv'][2]
        saturation = color_data['hsv'][1]

        # Water = very bright and low saturation (white/clear)
        return brightness >= WHITE_THRESHOLD and saturation < 30

    def find_matching_bottle(self, color_data):
        """
        Find the closest matching bottle from reference chart

        Args:
            color_data: Sample color data

        Returns:
            Tuple of (matched_bottle, confidence)
        """
        if not self.bottles:
            return None, 0.0

        min_distance = float('inf')
        matched = None

        for bottle in self.bottles:
            dist = self.color_distance(color_data, bottle.color_data)
            if dist < min_distance:
                min_distance = dist
                matched = bottle

        # Calculate confidence
        confidence = max(0, min(100, (1 - min_distance / MAX_COLOR_DISTANCE) * 100))

        return matched, confidence

    def color_distance(self, color1, color2):
        """
        Calculate perceptual color distance

        Args:
            color1, color2: Color data dictionaries

        Returns:
            Distance value (lower = more similar)
        """
        # HSV distance (weighted)
        h1, s1, v1 = color1['hsv']
        h2, s2, v2 = color2['hsv']
        hsv_dist = np.sqrt(
            (h2 - h1)**2 * HUE_WEIGHT +
            (s2 - s1)**2 * SATURATION_WEIGHT +
            (v2 - v1)**2 * VALUE_WEIGHT
        )

        # LAB distance (perceptually uniform)
        l1, a1, b1 = color1['lab']
        l2, a2, b2 = color2['lab']
        lab_dist = np.sqrt((l2 - l1)**2 + (a2 - a1)**2 + (b2 - b1)**2)

        # Combined distance
        return hsv_dist * HSV_WEIGHT + lab_dist * LAB_WEIGHT

    def create_gradient_bar(self):
        """Create gradient bar from bottle colors for visualization"""
        if not self.bottles:
            return

        gradient = np.zeros((GRADIENT_BAR_HEIGHT, GRADIENT_BAR_WIDTH, 3), dtype=np.uint8)

        # Create smooth gradient by interpolating between bottle colors
        for x in range(GRADIENT_BAR_WIDTH):
            # Map x position to brightness value (0-255)
            # Left = 255 (bright), Right = 0 (dark)
            brightness = 255 - (x / GRADIENT_BAR_WIDTH * 255)

            # Find two nearest bottles for interpolation
            closest_bottles = sorted(
                self.bottles,
                key=lambda b: abs(b.brightness - brightness)
            )[:2]

            if len(closest_bottles) >= 2:
                b1 = closest_bottles[0]
                b2 = closest_bottles[1]

                # Calculate interpolation weight
                v1 = b1.brightness
                v2 = b2.brightness

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

    def get_gradient_bar(self):
        """Get the gradient bar image"""
        return self.gradient_bar
