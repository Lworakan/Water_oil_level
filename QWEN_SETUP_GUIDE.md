# Qwen3-VL Model Setup Guide

This guide explains how to set up the Qwen3-VL model for enhanced oil detection.

## Important Note

**The Qwen model is OPTIONAL.** The system works perfectly with SAM2 alone. Only set up Qwen if you want additional object detection capabilities.

## Option 1: Use SAM2 Only (No Setup Required)

Simply use `use_qwen=False` in your code:

```python
detector = OilLevelDetector()
results = detector.process_image("image.jpg", use_qwen=False)
```

This is the **recommended approach** for most use cases.

## Option 2: Install Ollama (Easiest)

### Step 1: Install Ollama

**macOS/Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows:**
Download from: https://ollama.com/download

### Step 2: Pull a Vision Model

```bash
# Try Qwen2-VL (vision model)
ollama pull qwen2-vl:2b

# Or LLaVA (alternative vision model)
ollama pull llava:7b
```

### Step 3: Update the Model Name

Edit `oil_level_detector.py` line 55:

```python
ollama_command = [
    "ollama", "run",
    "qwen2-vl:2b",  # Change this to your model name
    prompt,
    abs_image_path
]
```

### Step 4: Run with Qwen

```python
detector = OilLevelDetector()
results = detector.process_image("image.jpg", use_qwen=True)
```

## Option 3: Use Docker (Advanced)

### Prerequisites
- Docker installed and running
- Sufficient disk space (~5GB)

### Step 1: Pull Docker Image

The Docker implementation in the code will attempt to use a Hugging Face model. However, this requires a properly configured Docker image.

**Note:** The Docker approach is more complex and may require custom Docker image setup.

## Troubleshooting

### Error: "Model not found"

**Solution:** Make sure you've pulled the correct model name:

```bash
ollama list  # Check installed models
ollama pull qwen2-vl:2b  # Pull the model
```

### Error: "Ollama command failed"

**Solution 1:** Check if Ollama is running:

```bash
ollama serve  # Start Ollama server
```

**Solution 2:** Use SAM2 only:

```python
results = detector.process_image("image.jpg", use_qwen=False)
```

### Error: "Docker command failed"

**Solution:** Use Ollama instead, or disable Qwen:

```python
results = detector.process_image("image.jpg", use_qwen=False)
```

## Model Recommendations

For oil level detection, these models work well:

1. **qwen2-vl:2b** - Good balance of speed and accuracy
2. **llava:7b** - Higher accuracy, slower
3. **llava:13b** - Best accuracy, slowest

## Expected Output Format

When Qwen detection is successful, you'll get:

```json
{
  "bounding_boxes": [
    {
      "class": "used_cooking_oil",
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.95
    },
    {
      "class": "empty_space",
      "bbox": [x1, y1, x2, y2],
      "confidence": 0.92
    }
  ],
  "percent_by_area": {
    "used_cooking_oil": 37.8,
    "empty_space": 62.2
  }
}
```

## Quick Setup Script

Run the setup script:

```bash
chmod +x setup_qwen_model.sh
./setup_qwen_model.sh
```

## Still Having Issues?

Just use SAM2 only! It provides excellent results without the complexity of setting up additional models:

```bash
python oil_level_detector.py your_image.jpg
# By default, use_qwen is set to True in the file, change it to False
```

Or edit the `main()` function in `oil_level_detector.py`:

```python
results = detector.process_image(
    image_path,
    use_qwen=False,  # Disable Qwen
    output_path="oil_detection_result.jpg"
)
```
