import os
import sys
import io
import unittest
import json
from unittest.mock import patch
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import create_app
from app.services.result_manager import result_manager


class TestFrontendBackendIntegrationFast(unittest.TestCase):
    """
    Lightning-fast verification of all 3.12 endpoints, CORS, payload validation,
    regeneration, history persistence, and deletion with mocked model forward passes.
    """

    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.app.config["TESTING"] = True
        cls.client = cls.app.test_client()

        # Valid test image bytes
        img = Image.new("RGB", (320, 240), color=(120, 160, 200))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        cls.sample_image = buf.getvalue()

    def setUp(self):
        result_manager.clear_all()

    @patch("app.routes.analyze_image_semantics")
    @patch("app.routes.generate_caption")
    def test_3_12_1_and_3_12_2_upload_and_caption_generation(self, mock_gen, mock_sem):
        """3.12.1 & 3.12.2: Test browser upload and caption generation API connection."""
        mock_gen.return_value = {
            "caption": "A scenic view of a blue sky over clear water.",
            "captions": [{"id": 1, "caption": "A scenic view of a blue sky over clear water.", "confidence": {"score": 93.5, "level": "High"}, "text_stats": {"word_count": 10, "char_count": 48}}],
            "style": "short",
            "confidence": {"score": 93.5, "level": "High"},
            "text_stats": {"word_count": 10, "char_count": 48},
            "inference_time_ms": 240.5,
            "model_metadata": {"model": "Salesforce/blip-image-captioning-base", "device": "cpu", "style": "short"}
        }
        mock_sem.return_value = {
            "objects": ["Sky", "Water"],
            "scene": "Outdoor nature",
            "activity": "Calm",
            "attributes": ["Clear", "Blue"]
        }

        img_file = (io.BytesIO(self.sample_image), "sample_photo.jpg")
        res = self.client.post(
            "/api/caption",
            data={"image": img_file, "style": "short", "count": "1"},
            content_type="multipart/form-data"
        )
        self.assertEqual(res.status_code, 200)
        data = res.get_json()

        # 3.12.4: Verify all frontend required attributes
        self.assertTrue(data["success"])
        self.assertIn("id", data)
        self.assertIn("caption", data)
        self.assertIn("confidence", data)
        self.assertIn("semantics", data)
        self.assertIn("quality_evaluation", data)
        self.assertIn("image_metadata", data)
        self.assertIn("inference_time_ms", data)
        self.assertEqual(data["style"], "short")
        print(f"3.12.1 & 3.12.2 -> POST /api/caption succeeded, ID: {data['id']}")

    def test_3_12_5_and_3_12_10_api_error_handling(self):
        """3.12.5 & 3.12.10: Verify frontend error handling (400, 413, 404)."""
        # Missing file
        res_no_img = self.client.post("/api/caption", data={})
        self.assertEqual(res_no_img.status_code, 400)
        self.assertFalse(res_no_img.get_json()["success"])

        # Oversized file (> 10MB)
        oversized = (io.BytesIO(b"0" * (11 * 1024 * 1024)), "huge.jpg")
        res_large = self.client.post(
            "/api/caption",
            data={"image": oversized},
            content_type="multipart/form-data"
        )
        self.assertEqual(res_large.status_code, 413)
        self.assertIn("exceeds maximum", res_large.get_json()["message"].lower())

        # 404 endpoint
        res_404 = self.client.get("/api/caption/non_existent_id")
        self.assertEqual(res_404.status_code, 404)
        print("3.12.5 & 3.12.10 -> Error handling (400, 413, 404) verified successfully.")

    @patch("app.routes.analyze_image_semantics")
    @patch("app.routes.generate_caption")
    def test_3_12_6_7_8_history_and_deletion(self, mock_gen, mock_sem):
        """3.12.6, 3.12.7, 3.12.8: Test dynamic history loading, retrieval, and deletion."""
        mock_gen.return_value = {
            "caption": "A cute cat resting indoors.",
            "captions": [{"id": 1, "caption": "A cute cat resting indoors.", "confidence": {"score": 95.0, "level": "High"}, "text_stats": {"word_count": 5, "char_count": 27}}],
            "style": "standard",
            "confidence": {"score": 95.0, "level": "High"},
            "text_stats": {"word_count": 5, "char_count": 27},
            "inference_time_ms": 180.0,
            "model_metadata": {"model": "Salesforce/blip-image-captioning-base"}
        }
        mock_sem.return_value = {"objects": ["Cat"], "scene": "Indoor"}

        # 1. Generate record
        img_file = (io.BytesIO(self.sample_image), "cat.jpg")
        post_res = self.client.post(
            "/api/caption",
            data={"image": img_file, "style": "standard"},
            content_type="multipart/form-data"
        )
        rec_id = post_res.get_json()["id"]

        # 2. Dynamic history fetch
        hist_res = self.client.get("/api/history")
        self.assertEqual(hist_res.status_code, 200)
        self.assertEqual(hist_res.get_json()["data"]["total"], 1)

        # 3. Delete item
        del_res = self.client.delete(f"/api/history/{rec_id}")
        self.assertEqual(del_res.status_code, 200)
        self.assertEqual(self.client.get("/api/history").get_json()["data"]["total"], 0)
        print("3.12.6, 3.12.7, 3.12.8 -> Dynamic history load and deletion verified successfully.")

    @patch("app.routes.generate_caption")
    def test_3_12_9_caption_regeneration(self, mock_gen):
        """3.12.9: Test UI-triggered caption regeneration using result_id."""
        mock_gen.return_value = {
            "caption": "An artistic creative variation of the image.",
            "captions": [{"id": 1, "caption": "An artistic creative variation of the image.", "confidence": {"score": 91.0, "level": "High"}, "text_stats": {"word_count": 7, "char_count": 44}}],
            "style": "creative",
            "confidence": {"score": 91.0, "level": "High"},
            "text_stats": {"word_count": 7, "char_count": 44},
            "inference_time_ms": 210.0,
            "model_metadata": {"model": "Salesforce/blip-image-captioning-base"}
        }

        # Seed initial record
        saved = result_manager.save_result(
            caption="Initial caption.",
            captions=[],
            style="standard",
            image_metadata={"filename": "mock.jpg", "width": 100, "height": 100},
            confidence={"score": 90.0, "level": "High"},
            text_stats={"word_count": 2},
            model_metadata={},
            inference_time_ms=100.0
        )
        rec_id = saved["id"]

        # Create dummy file so os.path.exists passes
        upload_dir = self.app.config["UPLOAD_FOLDER"]
        os.makedirs(upload_dir, exist_ok=True)
        dummy_file = os.path.join(upload_dir, "mock.jpg")
        with open(dummy_file, "wb") as f:
            f.write(self.sample_image)

        try:
            regen_res = self.client.post(
                "/api/caption/regenerate",
                data=json.dumps({"result_id": rec_id, "style": "creative"}),
                content_type="application/json"
            )
            self.assertEqual(regen_res.status_code, 200)
            data = regen_res.get_json()
            self.assertTrue(data["success"])
            self.assertTrue(data["is_regenerated"])
            self.assertEqual(data["caption"], "An artistic creative variation of the image.")
            print("3.12.9 -> Caption regeneration endpoint verified successfully.")
        finally:
            if os.path.exists(dummy_file):
                os.remove(dummy_file)

    def test_3_12_12_cors_headers(self):
        """3.12.12: Verify universal CORS headers on all endpoints."""
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.headers.get("Access-Control-Allow-Origin"), "*")
        print("3.12.12 -> Universal CORS header verified.")


if __name__ == "__main__":
    unittest.main()
