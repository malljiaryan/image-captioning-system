import io
import os
import sys
from PIL import Image

# Ensure backend root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import create_app
from app.ai.caption_generator import generate_caption

def get_test_image():
    buf = io.BytesIO()
    # Create an image with visual contrast: blue ocean and sunny beach
    img = Image.new("RGB", (320, 240), color=(0, 105, 148))
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf

def run_advanced_tests():
    print("==========================================================")
    print("  Milestone 3.8: Advanced Captioning Features Test Suite  ")
    print("==========================================================")

    # 1. Short Style Test
    print("\n[Test 1] Short Caption Mode")
    res_short = generate_caption(get_test_image(), style="short", num_captions=1)
    print(f"  -> Short Caption: '{res_short['caption']}'")
    print(f"  -> Word count   : {res_short['text_stats']['word_count']}")
    assert res_short["style"] == "short"
    assert res_short["text_stats"]["word_count"] <= 15

    # 2. Detailed Style Test
    print("\n[Test 2] Detailed Caption Mode")
    res_detailed = generate_caption(get_test_image(), style="detailed", num_captions=1)
    print(f"  -> Detailed Caption: '{res_detailed['caption']}'")
    print(f"  -> Word count      : {res_detailed['text_stats']['word_count']}")
    assert res_detailed["style"] == "detailed"
    assert res_detailed["text_stats"]["word_count"] >= 5

    # 3. Creative Style Test (Sampling)
    print("\n[Test 3] Creative Caption Mode")
    res_creative = generate_caption(get_test_image(), style="creative", num_captions=1)
    print(f"  -> Creative Caption: '{res_creative['caption']}'")
    print(f"  -> Decoding method : {res_creative['model_metadata']['decoding_method']}")
    assert res_creative["style"] == "creative"
    assert res_creative["model_metadata"]["decoding_method"] == "nucleus_sampling"

    # 4. Multiple Captions Test (3 Variations)
    print("\n[Test 4] Multiple Caption Generation (3 Variations)")
    res_multi = generate_caption(get_test_image(), style="standard", num_captions=3)
    print(f"  -> Total variations: {len(res_multi['captions'])}")
    for c in res_multi["captions"]:
        print(f"     Variation {c['id']}: '{c['caption']}' (Confidence: {c['confidence']['score']}%)")
    assert len(res_multi["captions"]) == 3

    # 5. Full Flask API Integration Test
    print("\n[Test 5] Flask API POST /api/caption (style=detailed, count=3)")
    app = create_app()
    app.config["TESTING"] = True
    client = app.test_client()

    api_res = client.post("/api/caption", data={
        "image": (get_test_image(), "ocean_beach.jpg"),
        "style": "detailed",
        "count": "3"
    }, content_type="multipart/form-data")

    assert api_res.status_code == 200
    data = api_res.get_json()
    assert data["success"] is True
    assert data["style"] == "detailed"
    assert len(data["captions"]) == 3
    print(f"  -> API Style    : {data['style']}")
    print(f"  -> Variations   : {len(data['captions'])} returned successfully")
    print(f"  -> Primary      : '{data['caption']}'")

    print("\n[SUCCESS] All advanced captioning feature tests passed successfully!")


if __name__ == "__main__":
    run_advanced_tests()
