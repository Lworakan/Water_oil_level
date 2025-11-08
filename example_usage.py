"""
Example usage of the Oil Level Detection System
Demonstrates both SAM2-only and Qwen+SAM2 workflows
"""

from oil_level_detector import OilLevelDetector
import json
import os


def example_1_sam2_only():
    """Example 1: Using SAM2 segmentation only (no Docker required)"""
    print("="*60)
    print("Example 1: SAM2-Only Detection")
    print("="*60)

    # Initialize detector
    detector = OilLevelDetector(sam_model_path="sam2.1_b.pt")

    # Process image
    image_path = "screenshots/screenshot_20251108_210827.jpg"

    if os.path.exists(image_path):
        results = detector.process_image(
            image_path,
            use_qwen=False,
            output_path="example1_result.jpg"
        )

        print("\n✓ Generated files:")
        print("  - example1_result.jpg (visualization)")
        print("  - example1_result_results.json (JSON data)")
        print(f"\n✓ Oil level: {results['percent_by_area']['used_cooking_oil']}%")
        print(f"✓ Empty space: {results['percent_by_area']['empty_space']}%")
    else:
        print(f"Error: Image not found: {image_path}")

    print()


def example_2_qwen_and_sam2():
    """Example 2: Using Qwen3-VL + SAM2 (requires Docker)"""
    print("="*60)
    print("Example 2: Qwen3-VL + SAM2 Detection")
    print("="*60)
    print("Note: This requires Docker and the Qwen model")
    print()

    # Initialize detector
    detector = OilLevelDetector(sam_model_path="sam2.1_b.pt")

    # Process image with Qwen detection
    image_path = "screenshots/screenshot_20251108_210827.jpg"

    if os.path.exists(image_path):
        results = detector.process_image(
            image_path,
            use_qwen=True,  # Enable Qwen detection
            output_path="example2_result.jpg"
        )

        print("\n✓ Generated files:")
        print("  - example2_result.jpg (SAM2 visualization)")
        print("  - example2_result_qwen_visualization.jpg (Qwen visualization)")
        print("  - example2_result_qwen_results.json (Qwen JSON)")
        print("  - example2_result_results.json (Final JSON)")
        print(f"\n✓ Oil level: {results['percent_by_area']['used_cooking_oil']}%")
        print(f"✓ Empty space: {results['percent_by_area']['empty_space']}%")
    else:
        print(f"Error: Image not found: {image_path}")

    print()


def example_3_custom_image():
    """Example 3: Process your own image"""
    print("="*60)
    print("Example 3: Process Custom Image")
    print("="*60)

    # Get image path from user
    image_path = input("Enter image path (or press Enter to skip): ").strip()

    if not image_path or not os.path.exists(image_path):
        print("Skipping custom image example")
        return

    # Initialize detector
    detector = OilLevelDetector(sam_model_path="sam2.1_b.pt")

    # Process image
    results = detector.process_image(
        image_path,
        use_qwen=False,
        output_path="custom_result.jpg"
    )

    print("\n✓ Processing complete!")
    print(f"✓ Oil level: {results['percent_by_area']['used_cooking_oil']}%")
    print(f"✓ Empty space: {results['percent_by_area']['empty_space']}%")
    print()


def example_4_programmatic_access():
    """Example 4: Programmatic access to results"""
    print("="*60)
    print("Example 4: Programmatic Access")
    print("="*60)

    detector = OilLevelDetector(sam_model_path="sam2.1_b.pt")
    image_path = "screenshots/screenshot_20251108_210827.jpg"

    if os.path.exists(image_path):
        # Process image
        results = detector.process_image(
            image_path,
            use_qwen=False,
            output_path="example4_result.jpg"
        )

        # Access results programmatically
        oil_percentage = results["percent_by_area"]["used_cooking_oil"]
        empty_percentage = results["percent_by_area"]["empty_space"]

        print(f"\n✓ Results extracted:")
        print(f"  Oil: {oil_percentage}%")
        print(f"  Empty: {empty_percentage}%")

        # Make decisions based on results
        if oil_percentage > 80:
            print("\n⚠ Alert: Bottle is almost full!")
        elif oil_percentage < 20:
            print("\n⚠ Alert: Bottle is almost empty!")
        else:
            print(f"\n✓ Bottle status: {oil_percentage}% full")

        # Export to different formats
        print("\n✓ Exporting to CSV...")
        with open("example4_results.csv", "w") as f:
            f.write("class,percentage\n")
            for class_name, percentage in results["percent_by_area"].items():
                f.write(f"{class_name},{percentage}\n")
        print("  - example4_results.csv created")
    else:
        print(f"Error: Image not found: {image_path}")

    print()


def main():
    """Run all examples"""
    print("\n")
    print("╔════════════════════════════════════════════════════════════╗")
    print("║        Oil Level Detection - Example Usage                ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print()

    # Run examples
    example_1_sam2_only()

    # Ask user if they want to run Qwen example
    run_qwen = input("Run Qwen+SAM2 example? (requires Docker) [y/N]: ").lower()
    if run_qwen == 'y':
        example_2_qwen_and_sam2()

    # Run programmatic example
    example_4_programmatic_access()

    # Optional custom image
    run_custom = input("\nProcess a custom image? [y/N]: ").lower()
    if run_custom == 'y':
        example_3_custom_image()

    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60)


if __name__ == "__main__":
    main()
