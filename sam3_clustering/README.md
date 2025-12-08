# SAM3 with 2K Clustering Integration

This folder contains the integration of SAM3 segmentation with K-Means clustering (K=2) for analyzing object composition (e.g., oil vs water levels).

## Structure

- `main.py`: Main entry point script.
- `sam3_segmentation.py`: Contains the `Sam3Segmenter` class for handling SAM3 model operations.
- `clustering_utils.py`: Contains the `apply_clustering` function for K-Means clustering and visualization.
- `output/`: Directory where processed images will be saved.

## Usage

1.  Ensure you have the required dependencies installed:
    ```bash
    pip install torch transformers opencv-python scikit-learn pillow matplotlib
    ```

2.  Run the main script:
    ```bash
    python sam3_clustering/main.py
    ```

## Configuration

You can modify the `main()` function in `main.py` to change:
- `IMAGE_PATH`: Path to the input image.
- `TEXT_PROMPT`: The text prompt for SAM3 (default: "bottle").
- `N_CLUSTERS`: Number of clusters for K-Means (default: 2).
