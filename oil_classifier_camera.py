#!/usr/bin/env python3
"""
Webcam-based Oil Classification System
Uses reference color chart to classify oil condition (usable vs unusable)
"""

import cv2
import numpy as np
import json
import os
from datetime import datetime

# Configuration
REFERENCE_IMAGE = "unnamed.jpg"  # Oil color reference chart
CALIBRATION_FILE = "oil_calibration.json"
CAMERA_INDEX = 0  # Change if you have multiple cameras

class OilClassifier:
    def __init__(self, reference_image_path):
        self.reference_image_path = reference_image_path
        self.good_oil_colors = []  # Top row - acceptable oil
        self.bad_oil_colors = []   # Bottom row - unusable oil
        self.load_reference_chart()

    def load_reference_chart(self):
        """Load and analyze the reference oil color chart"""
        if not os.path.exists(self.reference_image_path):
            print(f"Warning: Reference image '{self.reference_image_path}' not found")
            print("Using default color ranges")
            self.use_default_ranges()
            return

        img = cv2.imread(self.reference_image_path)
        if img is None:
            print("Error loading reference image")
            self.use_default_ranges()
            return

        h, w = img.shape[:2]

        # Extract top row (good oil) - approximately upper 45% of image
        good_row = img[int(h * 0.15):int(h * 0.45), :]

        # Extract bottom row (bad oil) - approximately lower 45% of image
        bad_row = img[int(h * 0.55):int(h * 0.85), :]

        # Sample colors from multiple bottles in each row
        self.good_oil_colors = self.extract_colors_from_row(good_row, num_samples=10)
        self.bad_oil_colors = self.extract_colors_from_row(bad_row, num_samples=10)

        print(f"Loaded reference chart: {len(self.good_oil_colors)} good samples, {len(self.bad_oil_colors)} bad samples")

    def extract_colors_from_row(self, row_img, num_samples=10):
        """Extract average colors from bottles in a row"""
        h, w = row_img.shape[:2]
        colors = []

        # Divide row into sections (one per bottle)
        section_width = w // num_samples

        for i in range(num_samples):
            x1 = i * section_width + section_width // 4
            x2 = (i + 1) * section_width - section_width // 4
            y1 = h // 4
            y2 = 3 * h // 4

            # Extract center region of each bottle
            sample = row_img[y1:y2, x1:x2]

            # Convert to HSV and LAB for better color analysis
            hsv = cv2.cvtColor(sample, cv2.COLOR_BGR2HSV)
            lab = cv2.cvtColor(sample, cv2.COLOR_BGR2LAB)

            # Calculate average color values
            avg_hue = np.mean(hsv[:, :, 0])
            avg_sat = np.mean(hsv[:, :, 1])
            avg_val = np.mean(hsv[:, :, 2])
            avg_l = np.mean(lab[:, :, 0])
            avg_a = np.mean(lab[:, :, 1])
            avg_b = np.mean(lab[:, :, 2])

            colors.append({
                'hsv': (avg_hue, avg_sat, avg_val),
                'lab': (avg_l, avg_a, avg_b)
            })

        return colors

    def use_default_ranges(self):
        """Use default color ranges if reference image is not available"""
        # Good oil: Light yellow to medium brown
        for i in range(10):
            val = 180 - (i * 15)  # Decreasing brightness
            sat = 100 + (i * 10)  # Increasing saturation
            self.good_oil_colors.append({
                'hsv': (15 + i * 2, sat, val),
                'lab': (val * 0.4, 128, 128 + i * 5)
            })

        # Bad oil: Dark brown to black
        for i in range(10):
            val = 100 - (i * 10)  # Very dark
            self.bad_oil_colors.append({
                'hsv': (10, 80 + i * 5, val),
                'lab': (val * 0.4, 128, 125)
            })

    def analyze_oil_sample(self, frame, roi=None):
        """
        Analyze oil color from camera frame
        Returns: dict with color metrics
        """
        if roi is None:
            # Use center 50% of frame as ROI
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

        return {
            'hsv': (avg_hue, avg_sat, avg_val),
            'lab': (avg_l, avg_a, avg_b)
        }

    def color_distance(self, color1, color2):
        """Calculate color distance using both HSV and LAB"""
        # HSV distance (weighted)
        h1, s1, v1 = color1['hsv']
        h2, s2, v2 = color2['hsv']
        hsv_dist = np.sqrt((h2 - h1)**2 * 0.5 + (s2 - s1)**2 * 0.3 + (v2 - v1)**2 * 1.0)

        # LAB distance (perceptually uniform)
        l1, a1, b1 = color1['lab']
        l2, a2, b2 = color2['lab']
        lab_dist = np.sqrt((l2 - l1)**2 + (a2 - a1)**2 + (b2 - b1)**2)

        # Combined distance
        return hsv_dist * 0.3 + lab_dist * 0.7

    def classify(self, frame):
        """
        Classify oil sample as usable or unusable based on color matching
        Returns: (classification, confidence, metrics, closest_match_index)
        """
        if not self.good_oil_colors or not self.bad_oil_colors:
            return "NO REFERENCE", 0.0, None, -1

        current = self.analyze_oil_sample(frame)

        # Find minimum distance to good oil samples
        good_distances = [self.color_distance(current, ref) for ref in self.good_oil_colors]
        min_good_dist = min(good_distances)
        closest_good_idx = good_distances.index(min_good_dist)

        # Find minimum distance to bad oil samples
        bad_distances = [self.color_distance(current, ref) for ref in self.bad_oil_colors]
        min_bad_dist = min(bad_distances)
        closest_bad_idx = bad_distances.index(min_bad_dist)

        # Classify based on nearest neighbor
        total_dist = min_good_dist + min_bad_dist
        if total_dist == 0:
            confidence = 50.0
            classification = "UNCERTAIN"
            closest_idx = -1
        elif min_good_dist < min_bad_dist:
            classification = "USABLE OIL"
            confidence = (min_bad_dist / total_dist) * 100
            closest_idx = closest_good_idx
        else:
            classification = "UNUSABLE OIL"
            confidence = (min_good_dist / total_dist) * 100
            closest_idx = closest_bad_idx + 10  # Offset for bad oil indices

        # Add darkness level (0-10 scale)
        darkness_level = int((255 - current['hsv'][2]) / 25.5)

        metrics = {
            'hue': current['hsv'][0],
            'saturation': current['hsv'][1],
            'value': current['hsv'][2],
            'darkness_level': darkness_level,
            'good_dist': min_good_dist,
            'bad_dist': min_bad_dist
        }

        return classification, confidence, metrics, closest_idx

def draw_roi(frame):
    """Draw region of interest rectangle on frame"""
    h, w = frame.shape[:2]
    x1, y1 = w // 4, h // 4
    x2, y2 = 3 * w // 4, 3 * h // 4
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, "Place oil sample here", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

def draw_reference_chart(frame, classifier):
    """Draw mini reference chart on the side"""
    h, w = frame.shape[:2]

    # Position for reference chart
    ref_x = w - 150
    ref_y = 10

    cv2.putText(frame, "Reference:", (ref_x, ref_y + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Draw good oil color boxes
    cv2.putText(frame, "Good", (ref_x, ref_y + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
    for i in range(min(5, len(classifier.good_oil_colors))):
        hsv = classifier.good_oil_colors[i]['hsv']
        color_bgr = cv2.cvtColor(np.uint8([[[hsv[0], hsv[1], hsv[2]]]]), cv2.COLOR_HSV2BGR)[0][0]
        color_bgr = tuple(map(int, color_bgr))
        cv2.rectangle(frame, (ref_x + i * 25, ref_y + 40),
                     (ref_x + i * 25 + 20, ref_y + 60), color_bgr, -1)
        cv2.rectangle(frame, (ref_x + i * 25, ref_y + 40),
                     (ref_x + i * 25 + 20, ref_y + 60), (200, 200, 200), 1)

    # Draw bad oil color boxes
    cv2.putText(frame, "Bad", (ref_x, ref_y + 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
    for i in range(min(5, len(classifier.bad_oil_colors))):
        hsv = classifier.bad_oil_colors[i]['hsv']
        color_bgr = cv2.cvtColor(np.uint8([[[hsv[0], hsv[1], hsv[2]]]]), cv2.COLOR_HSV2BGR)[0][0]
        color_bgr = tuple(map(int, color_bgr))
        cv2.rectangle(frame, (ref_x + i * 25, ref_y + 85),
                     (ref_x + i * 25 + 20, ref_y + 105), color_bgr, -1)
        cv2.rectangle(frame, (ref_x + i * 25, ref_y + 85),
                     (ref_x + i * 25 + 20, ref_y + 105), (200, 200, 200), 1)

def main():
    print("="*60)
    print("Webcam Oil Classification System - Color Pattern Matching")
    print("="*60)
    print(f"\nUsing reference image: {REFERENCE_IMAGE}")
    print("\nControls:")
    print("  'c' - Toggle continuous classification")
    print("  'r' - Show/hide reference chart")
    print("  'q' - Quit")
    print("="*60)

    # Initialize camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Error: Cannot open camera {CAMERA_INDEX}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    classifier = OilClassifier(REFERENCE_IMAGE)
    continuous_mode = True
    show_reference = True

    print("\nCamera ready. Place oil sample in the green box.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read frame")
            break

        # Draw ROI
        draw_roi(frame)

        # Draw reference chart
        if show_reference:
            draw_reference_chart(frame, classifier)

        # Classification
        if continuous_mode:
            classification, confidence, metrics, closest_idx = classifier.classify(frame)

            # Color coding
            if classification == "USABLE OIL":
                color = (0, 255, 0)  # Green
            elif classification == "UNUSABLE OIL":
                color = (0, 0, 255)  # Red
            else:
                color = (0, 165, 255)  # Orange

            # Display classification
            cv2.putText(frame, f"{classification}", (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            cv2.putText(frame, f"Confidence: {confidence:.1f}%", (10, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            if metrics:
                cv2.putText(frame, f"Darkness: {metrics['darkness_level']}/10", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"Value: {metrics['value']:.0f}", (10, 135),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow('Oil Classifier - Pattern Matching', frame)

        # Handle keypresses
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('c'):
            continuous_mode = not continuous_mode
            status = "enabled" if continuous_mode else "disabled"
            print(f"Continuous classification {status}")
        elif key == ord('r'):
            show_reference = not show_reference
            print(f"Reference chart {'shown' if show_reference else 'hidden'}")

    cap.release()
    cv2.destroyAllWindows()
    print("\nProgram terminated")

if __name__ == "__main__":
    main()
