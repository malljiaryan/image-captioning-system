import os
import sys
from PIL import Image, ImageDraw

# Ensure backend root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.ai.caption_generator import generate_caption

def create_sample_images():
    os.makedirs("tests/sample_images", exist_ok=True)

    # 1. Landscape: Blue sky with green grass and bright yellow sun
    landscape = Image.new("RGB", (300, 200), color=(135, 206, 235))
    draw = ImageDraw.Draw(landscape)
    draw.rectangle([0, 140, 300, 200], fill=(34, 139, 34))  # green grass
    draw.ellipse([220, 20, 280, 80], fill=(255, 223, 0))   # yellow sun
    path1 = "tests/sample_images/landscape.jpg"
    landscape.save(path1)

    # 2. Red ball on dark table
    ball = Image.new("RGB", (300, 200), color=(50, 50, 50))
    draw = ImageDraw.Draw(ball)
    draw.ellipse([100, 50, 200, 150], fill=(220, 20, 60))
    path2 = "tests/sample_images/red_ball.png"
    ball.save(path2)

    return [path1, path2]


def run_tests():
    print("==================================================")
    print("  Milestone 3.6: Caption Generator Test Suite    ")
    print("==================================================")

    test_images = create_sample_images()
    
    # Also add existing uploaded document image if present
    doc_path = "uploads/New_Doc_06-24-2026_17.17_page-0001.jpg"
    if os.path.exists(doc_path):
        test_images.insert(0, doc_path)

    for i, img_path in enumerate(test_images, 1):
        print(f"\n[Test {i}/{len(test_images)}] Processing: {img_path}")
        result = generate_caption(img_path)

        print(f"  -> Caption        : {result['caption']}")
        print(f"  -> Inference Time : {result['inference_time_ms']} ms")
        print(f"  -> Model Used     : {result['model']}")
        print(f"  -> Device         : {result['device']}")

        assert isinstance(result["caption"], str) and len(result["caption"]) > 0, "Caption must be non-empty"
        assert result["inference_time_ms"] > 0, "Inference time must be positive"
        assert result["model"] == "Salesforce/blip-image-captioning-base"

    print("\n[SUCCESS] All caption generator tests passed successfully!")


if __name__ == "__main__":
    run_tests()
