import io
import os
import sys
from PIL import Image

# Ensure backend root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import create_app

def run_api_tests():
    print("==================================================")
    print("      Milestone 3.6: Flask API Test Suite         ")
    print("==================================================")

    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()

    # 1. Health check endpoint test
    print("\n[Test 1] GET /api/health")
    res = client.get("/api/health")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    health_data = res.get_json()
    assert health_data["status"] == "healthy"
    print("  -> Health check response:", health_data)

    # 2. Missing image payload test
    print("\n[Test 2] POST /api/caption (No image provided)")
    res = client.post("/api/caption", data={}, content_type="multipart/form-data")
    assert res.status_code == 400
    missing_data = res.get_json()
    assert missing_data["success"] is False
    print("  -> Error response:", missing_data["message"])

    # 3. Invalid extension test
    print("\n[Test 3] POST /api/caption (Unsupported file type .txt)")
    data = {
        "image": (io.BytesIO(b"this is not an image"), "notes.txt")
    }
    res = client.post("/api/caption", data=data, content_type="multipart/form-data")
    assert res.status_code == 400
    ext_data = res.get_json()
    assert ext_data["success"] is False
    print("  -> Error response:", ext_data["message"])

    # 4. Valid image caption generation test
    print("\n[Test 4] POST /api/caption (Valid image upload)")
    # Create in-memory test image
    img_byte_arr = io.BytesIO()
    test_img = Image.new("RGB", (200, 200), color=(73, 109, 137))
    test_img.save(img_byte_arr, format="JPEG")
    img_byte_arr.seek(0)

    data = {
        "image": (img_byte_arr, "api_test_image.jpg")
    }
    res = client.post("/api/caption", data=data, content_type="multipart/form-data")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.get_data(as_text=True)}"
    caption_data = res.get_json()
    assert caption_data["success"] is True
    assert "caption" in caption_data and len(caption_data["caption"]) > 0
    print("  -> Generated Caption :", caption_data["caption"])
    print("  -> Inference Time    :", caption_data["inference_time_ms"], "ms")
    print("  -> Saved Filename    :", caption_data["filename"])

    print("\n[SUCCESS] All API integration tests passed successfully!")


if __name__ == "__main__":
    run_api_tests()
