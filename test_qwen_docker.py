"""
Quick test script for Docker model ai/qwen3-vl
Tests if the Docker model is working correctly
"""

import subprocess
import os
import sys


def test_docker_model():
    """Test if docker model run ai/qwen3-vl is working"""
    print("="*60)
    print("Testing Docker Model: ai/qwen3-vl")
    print("="*60)
    print()

    # Check if Docker is running
    print("1. Checking if Docker is running...")
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print("   ✓ Docker is running")
        else:
            print("   ✗ Docker is not running")
            print("   Please start Docker and try again")
            return False
    except Exception as e:
        print(f"   ✗ Error checking Docker: {e}")
        return False

    print()

    # Check if we have a test image
    print("2. Looking for test image...")
    test_images = [
        "screenshots/screenshot_20251108_210827.jpg",
        "result.jpg",
        "unnamed.jpg"
    ]

    test_image = None
    for img in test_images:
        if os.path.exists(img):
            test_image = img
            print(f"   ✓ Found test image: {test_image}")
            break

    if not test_image:
        print("   ✗ No test image found")
        print(f"   Searched for: {', '.join(test_images)}")
        return False

    print()

    # Test the docker model command
    print("3. Testing docker model run command...")
    print("   This may take a moment (downloading model if needed)...")
    print()

    abs_image_path = os.path.abspath(test_image)
    prompt = "Describe what you see in this image briefly."

    docker_command = [
        "docker", "model", "run",
        "ai/qwen3-vl",
        prompt,
        abs_image_path
    ]

    print(f"   Command: {' '.join(docker_command)}")
    print()

    try:
        result = subprocess.run(
            docker_command,
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode == 0:
            print("   ✓ Docker model command succeeded!")
            print()
            print("   Output:")
            print("   " + "-"*56)
            output_lines = result.stdout.strip().split('\n')
            for line in output_lines[:10]:  # Show first 10 lines
                print(f"   {line}")
            if len(output_lines) > 10:
                print(f"   ... ({len(output_lines) - 10} more lines)")
            print("   " + "-"*56)
            print()
            return True
        else:
            print(f"   ✗ Docker model command failed (return code: {result.returncode})")
            print()
            print("   Error output:")
            print("   " + "-"*56)
            print(f"   {result.stderr}")
            print("   " + "-"*56)
            print()
            return False

    except subprocess.TimeoutExpired:
        print("   ✗ Command timed out (>120 seconds)")
        return False
    except Exception as e:
        print(f"   ✗ Error running command: {e}")
        return False


def main():
    """Run the test"""
    success = test_docker_model()

    print()
    print("="*60)
    if success:
        print("✓ SUCCESS - Docker model is working!")
        print()
        print("You can now run:")
        print("  python oil_level_detector.py")
    else:
        print("✗ FAILED - Docker model is not working")
        print()
        print("Options:")
        print("  1. Check Docker Desktop is running")
        print("  2. Try: docker model pull ai/qwen3-vl")
        print("  3. Use SAM2 only (set use_qwen=False)")
    print("="*60)
    print()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
