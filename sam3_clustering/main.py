import os
import cv2
from sam3_segmentation import Sam3Segmenter
from clustering_utils import apply_clustering

def main():
    IMAGE_PATH = "/Users/worakanlasudee/Documents/PlatformIO/Projects/Water_oil_level/screenshots/screenshot_20251108_210827.jpg"
    OUTPUT_DIR = "sam3_clustering/output"
    TEXT_PROMPT = "bottle"
    N_CLUSTERS = 2
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    try:
        segmenter = Sam3Segmenter()
        
        original_img, masks = segmenter.segment_image(IMAGE_PATH, text_prompt=TEXT_PROMPT)
        
        if masks is not None and len(masks) > 0:
            result_img = apply_clustering(original_img, masks, n_clusters=N_CLUSTERS)
            
            output_filename = f"sam3_cluster_result_{os.path.basename(IMAGE_PATH)}"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            cv2.imwrite(output_path, result_img)
            print(f"\nProcessing complete. Result saved to: {output_path}")
        else:
            print("No objects found matching the prompt.")
            
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
