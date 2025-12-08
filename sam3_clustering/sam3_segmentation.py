import os
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import Sam3Processor, Sam3Model

class Sam3Segmenter:
    def __init__(self, model_name="facebook/sam3", device=None):
        """
        Initialize the SAM3 model and processor.
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        print(f"Loading SAM3 model: {model_name}...")
        try:
            self.model = Sam3Model.from_pretrained(model_name).to(self.device)
            self.processor = Sam3Processor.from_pretrained(model_name)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def segment_image(self, image_path, text_prompt="bottle"):
        """
        Segment the image using SAM3 with a text prompt.
        """
        print(f"Loading image: {image_path}")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        original_image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        print(f"Segmenting with prompt: '{text_prompt}'...")
        inputs = self.processor(images=image, text=text_prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=0.5,
            mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist()
        )[0]
        
        print(f"Found {len(results['masks'])} objects.")
        return original_image_cv, results['masks']
