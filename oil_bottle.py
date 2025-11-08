"""
Oil Bottle class and related utilities
"""

import cv2
import numpy as np
from config import *


class OilBottle:
    """Represents a reference bottle from the chart"""

    def __init__(self, x, y, w, h, color_data, is_usable, bottle_index):
        """
        Initialize a bottle

        Args:
            x, y: Position in reference image
            w, h: Width and height
            color_data: Dictionary with HSV, LAB, and BGR color data
            is_usable: True if from usable row, False if unusable
            bottle_index: Index number (0-19)
        """
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.color_data = color_data
        self.is_usable = is_usable
        self.bottle_index = bottle_index

    @property
    def brightness(self):
        """Get brightness value (0-255)"""
        return self.color_data['hsv'][2]

    @property
    def hue(self):
        """Get hue value"""
        return self.color_data['hsv'][0]

    @property
    def saturation(self):
        """Get saturation value"""
        return self.color_data['hsv'][1]

    def __repr__(self):
        status = "USABLE" if self.is_usable else "UNUSABLE"
        return f"OilBottle({self.bottle_index}, {status}, brightness={self.brightness:.1f})"


def extract_bottles_from_reference(reference_img):
    """
    Extract all bottles from reference image

    Args:
        reference_img: Reference chart image (BGR)

    Returns:
        List of OilBottle objects sorted by brightness (descending)
    """
    if reference_img is None:
        return []

    h, w = reference_img.shape[:2]

    # Top row (usable oil)
    top_row_y1 = int(h * TOP_ROW_Y_START)
    top_row_y2 = int(h * TOP_ROW_Y_END)

    # Bottom row (unusable oil)
    bottom_row_y1 = int(h * BOTTOM_ROW_Y_START)
    bottom_row_y2 = int(h * BOTTOM_ROW_Y_END)

    bottle_width = w // NUM_BOTTLES_PER_ROW

    all_bottles = []

    # Extract top row bottles (USABLE)
    for i in range(NUM_BOTTLES_PER_ROW):
        x1 = i * bottle_width + int(bottle_width * BOTTLE_HORIZONTAL_MARGIN)
        x2 = (i + 1) * bottle_width - int(bottle_width * BOTTLE_HORIZONTAL_MARGIN)
        y1 = top_row_y1 + int((top_row_y2 - top_row_y1) * BOTTLE_VERTICAL_MARGIN_TOP)
        y2 = top_row_y2 - int((top_row_y2 - top_row_y1) * BOTTLE_VERTICAL_MARGIN_BOTTOM)

        color_data = extract_color_from_region(reference_img, x1, y1, x2, y2)
        bottle = OilBottle(x1, y1, x2 - x1, y2 - y1, color_data,
                          is_usable=True, bottle_index=i)
        all_bottles.append(bottle)

    # Extract bottom row bottles (UNUSABLE)
    for i in range(NUM_BOTTLES_PER_ROW):
        x1 = i * bottle_width + int(bottle_width * BOTTLE_HORIZONTAL_MARGIN)
        x2 = (i + 1) * bottle_width - int(bottle_width * BOTTLE_HORIZONTAL_MARGIN)
        y1 = bottom_row_y1 + int((bottom_row_y2 - bottom_row_y1) * BOTTLE_VERTICAL_MARGIN_TOP)
        y2 = bottom_row_y2 - int((bottom_row_y2 - bottom_row_y1) * BOTTLE_VERTICAL_MARGIN_BOTTOM)

        color_data = extract_color_from_region(reference_img, x1, y1, x2, y2)
        bottle = OilBottle(x1, y1, x2 - x1, y2 - y1, color_data,
                          is_usable=False, bottle_index=i + NUM_BOTTLES_PER_ROW)
        all_bottles.append(bottle)

    # Sort by brightness (brightest first)
    bottles = sorted(all_bottles, key=lambda b: b.brightness, reverse=True)

    return bottles


def extract_color_from_region(img, x1, y1, x2, y2):
    """
    Extract average color from image region

    Args:
        img: BGR image
        x1, y1, x2, y2: Region coordinates

    Returns:
        Dictionary with 'hsv', 'lab', and 'bgr' color data
    """
    sample = img[y1:y2, x1:x2]

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
