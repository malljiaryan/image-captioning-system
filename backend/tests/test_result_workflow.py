import io
import os
import sys
from PIL import Image

# Ensure backend root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import create_app

def get_test_image_bytes():
    buf = io.BytesIO()
    img = Image.new("RGB", (320, 240), color=(100, 180, 255))
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf

def run_workflow_tests():
    print("==========================================================")
    print("  Milestone 3.7: Result Workflow & Metadata Test Suite    ")
    print("==========================================================")

    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()

    # 1. Standard caption generation request
    print("\n[Test 1] Standard Caption Request with Metadata")
    res = client.post("/api/caption", data={
        "image": (get_test_image_bytes(), "workflow_test.jpg")
    }, content_type="multipart/form-data")

    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    data = res.get_json()
    assert data["success"] is True

    print(f"  -> Caption         : {data['caption']}")
    print(f"  -> Confidence      : {data['confidence']['score']}% ({data['confidence']['level']})")
    print(f"  -> Text Stats      : {data['text_stats']['word_count']} words, {data['text_stats']['char_count']} chars")
    print(f"  -> Image Metadata  : {data['image_metadata']['width']}x{data['image_metadata']['height']}, {data['image_metadata']['format']}, {data['image_metadata']['size_kb']} KB")
    print(f"  -> Model Metadata  : {data['model_metadata']['model']} via {data['model_metadata']['decoding_method']}")
    print(f"  -> Inference Time  : {data['inference_time_ms']} ms")
    print(f"  -> Regenerated Flag: {data.get('is_regenerated', False)}")

    assert data["image_metadata"]["width"] == 320
    assert data["image_metadata"]["height"] == 240
    assert data["text_stats"]["word_count"] > 0
    assert data["confidence"]["score"] > 0
    assert data["is_regenerated"] is False

    # 2. Regenerate caption request (with regenerate=true flag)
    print("\n[Test 2] Regenerate Caption Request (Creative sampling)")
    res_regen = client.post("/api/caption", data={
        "image": (get_test_image_bytes(), "workflow_regen.jpg"),
        "regenerate": "true"
    }, content_type="multipart/form-data")

    assert res_regen.status_code == 200
    regen_data = res_regen.get_json()
    assert regen_data["success"] is True
    assert regen_data["is_regenerated"] is True
    print(f"  -> Regenerated Caption : {regen_data['caption']}")
    print(f"  -> Decoding Method     : {regen_data['model_metadata']['decoding_method']}")
    print(f"  -> Confidence          : {regen_data['confidence']['score']}%")

    print("\n[SUCCESS] All result workflow tests passed successfully!")


if __name__ == "__main__":
    run_workflow_tests()
