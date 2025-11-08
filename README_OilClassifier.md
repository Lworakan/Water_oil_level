# Oil Classification & Volume Estimation System

Professional Python application for classifying oil quality and estimating volume using computer vision and color analysis.

## Project Structure

```
Water_oil_level/
├── main.py                 # Application entry point
├── config.py              # Configuration settings
├── oil_bottle.py          # Bottle class and reference extraction
├── oil_detector.py        # Detection and classification logic
├── gui_components.py      # GUI components
├── unnamed.jpg            # Reference oil color chart
├── oil_classifier_gui.py  # (Legacy single-file version)
└── README_OilClassifier.md # This file
```

## Module Descriptions

### `main.py`
- Application entry point
- Initializes GUI and starts main loop
- Run with: `python main.py`

### `config.py`
- Central configuration file
- Camera settings (index, resolution)
- Color thresholds and detection parameters
- GUI dimensions and styling
- Container capacity settings

### `oil_bottle.py`
- `OilBottle` class: Represents a reference bottle from the chart
- `extract_bottles_from_reference()`: Extracts all 20 bottles from reference image
- `extract_color_from_region()`: Color extraction utility

### `oil_detector.py`
- `OilDetector` class: Main detection and analysis engine
- `analyze_sample()`: Analyzes oil color from camera frame
- `detect_oil_volume()`: **NEW IMPROVED ALGORITHM**
  - Separates oil pixels from empty space
  - White/bright pixels (≥200 brightness) = empty space
  - Colored pixels (yellow, orange, brown, dark) = oil
  - Volume = (oil_pixels / total_pixels) × container_capacity
- `find_matching_bottle()`: Matches sample to reference chart
- `color_distance()`: Perceptual color distance calculation
- `create_gradient_bar()`: Creates visual gradient scale

### `gui_components.py`
- `OilClassifierGUI` class: Main GUI application
- Camera feed display with ROI and oil level line
- Reference chart display with matched bottle highlighting
- Status indicators (USABLE/UNUSABLE/WATER)
- Gradient scale with brightness indicator
- Volume gauge with color-coded fill level
- Real-time classification and volume estimation

## Volume Calculation Algorithm

### Binary Preprocessing Method (Latest)

The system uses **binary image preprocessing** for accurate volume calculation:

1. **Extract ROI from camera frame**
2. **Convert to grayscale**
3. **Apply threshold:**
   ```python
   threshold = 200  # Configurable in config.py
   binary_mask = apply_threshold(gray, threshold, BINARY_INV)
   ```
   - Pixels < 200 (darker) → **White (255)** in binary = **OIL**
   - Pixels ≥ 200 (brighter) → **Black (0)** in binary = **EMPTY**

4. **Clean noise with morphological operations:**
   - Closing: Fill small holes in oil regions
   - Opening: Remove small noise pixels

5. **Count pixels:**
   ```
   oil_pixels = count(white pixels in binary mask)
   total_pixels = total pixels in ROI
   ```

6. **Calculate volume:**
   ```
   Fill Percentage = (oil_pixels / total_pixels) × 100%
   Volume (liters) = (Fill Percentage / 100) × Container Capacity
   ```

### Binary Image Display

The GUI shows the **preprocessed binary image** in real-time:
- **White pixels** = Oil detected
- **Black pixels** = Empty space
- Pixel counts displayed on the image
- Helps debug and verify detection accuracy

### Benefits:
- **Visual verification** of detection
- **Noise reduction** through morphological operations
- **Adjustable threshold** via `config.py`
- More accurate than simple level detection
- Handles irregular oil shapes
- Works with mixed oil colors
- Properly excludes water/empty space

### Tuning the Threshold

If volume detection is inaccurate, adjust `WHITE_THRESHOLD` in `config.py`:

```python
# Lower value = More sensitive (detects lighter oils)
WHITE_THRESHOLD = 180

# Higher value = Less sensitive (only detects darker oils)
WHITE_THRESHOLD = 220
```

**Watch the binary preview window** to see the effect!

## Features

✅ **Real-time oil quality classification**
- Compares sample to 20-bottle reference chart
- Classifies as USABLE or UNUSABLE
- Shows confidence percentage

✅ **Accurate volume estimation**
- Pixel-based volume calculation
- Separates oil from empty space
- 1-liter container support
- Visual fill gauge (0-100%)

✅ **Water/No-oil detection**
- Detects white/bright colors as empty
- Special status indicator for water

✅ **Visual feedback**
- Live camera feed with ROI
- **Binary preprocessing preview (NEW!)**
  - Shows exactly what the system sees
  - White = Oil detected
  - Black = Empty space
  - Pixel count overlay
- Oil level line indicator
- Bounding box on matching reference bottle
- Gradient brightness scale (0-255)
- Color-coded volume gauge

✅ **Professional code structure**
- Separated concerns (MVC-like pattern)
- Configurable parameters
- Reusable components
- Clean, documented code

## Installation

```bash
# Install dependencies
pip install opencv-python pillow numpy

# Run application
python main.py
```

## Usage

1. **Start the application:**
   ```bash
   python main.py
   ```

2. **Place oil sample in green detection area**

3. **System automatically:**
   - Classifies oil quality (usable/unusable)
   - Measures oil volume in liters
   - Shows fill percentage
   - Highlights matching bottle on reference chart

4. **Understanding the display:**
   - **Left Panel:**
     - Live camera feed with green ROI box
     - Binary preprocessing view below (white=oil, black=empty)
   - **Right Panel:**
     - Reference chart with matched bottle highlighted
     - Status indicators and metrics
   - **Status Colors:**
     - **Green** = Usable oil ✓
     - **Red** = Unusable oil ✗
     - **Blue** = Water/No oil detected
   - **Volume gauge colors:**
     - Green (75-100%) = High level
     - Yellow (50-75%) = Medium level
     - Orange (25-50%) = Medium-low level
     - Red (0-25%) = Low level

## Configuration

Edit `config.py` to customize:

```python
# Camera
CAMERA_INDEX = 0  # Change if multiple cameras

# Container
CONTAINER_CAPACITY_LITERS = 1.0  # Adjust for different sizes

# Thresholds
WHITE_THRESHOLD = 200  # Adjust for lighting conditions
```

## Technical Details

- **Color Spaces:** HSV and LAB for perceptual accuracy
- **Distance Metric:** Weighted HSV + LAB distance
- **Update Rate:** ~33 FPS (30ms per frame)
- **Resolution:** 640x480 camera feed
- **Reference Chart:** 2 rows × 10 bottles = 20 samples

## Senior Programming Practices Used

1. **Separation of Concerns:** Each module has a single responsibility
2. **Configuration Management:** Centralized config file
3. **Object-Oriented Design:** Encapsulated classes
4. **Code Documentation:** Docstrings and comments
5. **Modularity:** Reusable, testable components
6. **Error Handling:** Graceful fallbacks for missing files
7. **Clean Code:** Clear naming, organized structure

## Future Enhancements

- [ ] Support for different container sizes
- [ ] Calibration wizard for custom reference charts
- [ ] Data logging and history tracking
- [ ] Export results to CSV/JSON
- [ ] Multi-camera support
- [ ] Machine learning classification option

## License

Educational/Research Use

## Author

Senior Programmer - 2025
