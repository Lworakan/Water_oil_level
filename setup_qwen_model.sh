#!/bin/bash

# Setup script for Qwen3-VL model
# This script helps you set up either Ollama or Docker for running the Qwen model

echo "========================================="
echo "Qwen3-VL Model Setup"
echo "========================================="
echo ""

# Check if Ollama is installed
if command -v ollama &> /dev/null; then
    echo "✓ Ollama is installed"
    echo ""
    echo "Pulling Qwen3-VL model..."

    # Pull the model (note: adjust the model name if needed)
    # The actual model name might be different - check ollama.com/library
    ollama pull qwen2-vl:2b || echo "Note: Model name might need adjustment. Visit ollama.com/library for available models"

    echo ""
    echo "✓ Model setup complete!"
    echo ""
    echo "You can now run the detector with:"
    echo "  python oil_level_detector.py your_image.jpg"

else
    echo "✗ Ollama is not installed"
    echo ""
    echo "Option 1: Install Ollama (Recommended)"
    echo "  Visit: https://ollama.com/download"
    echo "  Or run: curl -fsSL https://ollama.com/install.sh | sh"
    echo ""
    echo "Option 2: Use Docker"
    echo "  Make sure Docker is installed and running"
    echo "  The script will use Docker automatically if Ollama is not available"
    echo ""
    echo "Option 3: Use SAM2 only (No Qwen required)"
    echo "  Set use_qwen=False in the script"
fi

echo ""
echo "========================================="
