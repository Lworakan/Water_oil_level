"""
Configuration settings for Oil Classification System
"""

# Camera settings
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Reference image
REFERENCE_IMAGE = "unnamed.jpg"

# Color thresholds
WHITE_THRESHOLD = 200  # Brightness above this = water/no oil
OIL_MIN_BRIGHTNESS = 0   # Minimum brightness for oil
OIL_MAX_BRIGHTNESS = 195  # Maximum brightness for oil (below white)

# Container specifications
CONTAINER_CAPACITY_LITERS = 1.0  # Total capacity in liters

# GUI settings
WINDOW_WIDTH = 1500
WINDOW_HEIGHT = 1000
WINDOW_TITLE = "Oil Classification & Volume Estimation System"

# Update rate
FRAME_UPDATE_MS = 30  # Update every 30ms (~33 FPS)

# Color distance calculation weights
HSV_WEIGHT = 0.3
LAB_WEIGHT = 0.7
HUE_WEIGHT = 0.5
SATURATION_WEIGHT = 0.3
VALUE_WEIGHT = 1.0

# Confidence calculation
MAX_COLOR_DISTANCE = 150.0

# Reference chart layout
NUM_BOTTLES_PER_ROW = 10
TOP_ROW_Y_START = 0.15  # Fraction of image height
TOP_ROW_Y_END = 0.45
BOTTOM_ROW_Y_START = 0.55
BOTTOM_ROW_Y_END = 0.85
BOTTLE_HORIZONTAL_MARGIN = 0.1  # Fraction of bottle width
BOTTLE_VERTICAL_MARGIN_TOP = 0.2  # Fraction of bottle height
BOTTLE_VERTICAL_MARGIN_BOTTOM = 0.1

# Gradient bar settings
GRADIENT_BAR_WIDTH = 512
GRADIENT_BAR_HEIGHT = 80
