import os
import sys
import io
import unittest
import json
from PIL import Image

# Ensure backend root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import create_app
from app.services.result_manager import result_manager
from app.ai.model import warmup_model


class TestBackendAPIIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialize Flask test client and ensure model is warm."""
        os.environ["TESTING"] = "true"
        cls.app = create_app()
        cls.app.config["TESTING"] = True
        cls.client = cls.app.test_client()

        # Warm up model for inference tests
        warmup_model()

        # Generate synthetic image for testing (300x200 RGB JPEG)
        img = Image.new("RGB", (300, 200), color=(70, 130, 180))
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format="JPEG")
        cls.valid_image_bytes = img_byte_arr.getvalue()

    def setUp(self):
        """Clean slate for history in tests if needed."""
        result_manager.clear_all()

    def test_3_11_1_and_3_11_2_app_structure_and_routes(self):
        """3.11.1 & 3.11.2: Verify Flask app structure and route organization."""
        self.assertIsNotNone(self.app)
        rules = [rule.rule for rule in self.app.url_map.iter_rules()]
        expected_routes = [
            "/",
            "/api/health",
            "/api/performance/stats",
            "/api/caption",
            "/api/caption/regenerate",
            "/api/caption/<result_id>",
            "/api/history",
            "/api/history/<result_id>",
            "/api/evaluate"
        ]
        for route in expected_routes:
            self.assertIn(route, rules, f"Expected route '{route}' should be registered.")

    def test_3_11_3_centralized_api_response_format(self):
        """3.11.3: Verify standard API response envelope format."""
        response = self.client.get("/api")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()

        self.assertIn("success", data)
        self.assertIn("message", data)
        self.assertIn("status_code", data)
        self.assertTrue(data["success"])
        self.assertEqual(data["status_code"], 200)

    def test_3_11_4_and_3_11_14_file_validation_missing_and_invalid_type(self):
        """3.11.4 & 3.11.14: Verify image upload validation for missing and invalid file types."""
        # 1. Missing file in request
        res1 = self.client.post("/api/caption", data={})
        self.assertEqual(res1.status_code, 400)
        d1 = res1.get_json()
        self.assertFalse(d1["success"])
        self.assertIn("No image file", d1["message"])

        # 2. Invalid file type (.txt file)
        fake_txt = (io.BytesIO(b"Not an image"), "test.txt")
        res2 = self.client.post(
            "/api/caption",
            data={"image": fake_txt},
            content_type="multipart/form-data"
        )
        self.assertEqual(res2.status_code, 400)
        d2 = res2.get_json()
        self.assertFalse(d2["success"])
        self.assertIn("Unsupported file format", d2["message"])

        # 3. Corrupted image file header
        fake_corrupted = (io.BytesIO(b"\xFF\xD8\xFFcorrupted data"), "corrupted.jpg")
        res3 = self.client.post(
            "/api/caption",
            data={"image": fake_corrupted},
            content_type="multipart/form-data"
        )
        self.assertEqual(res3.status_code, 400)
        d3 = res3.get_json()
        self.assertFalse(d3["success"])
        self.assertIn("Invalid or corrupted image", d3["message"])

    def test_3_11_13_file_size_validation(self):
        """3.11.13: Verify file size limit enforcement (reject > 10MB)."""
        # Create a mock file exceeding 10 MB (11 MB of zeros)
        large_bytes = b"0" * (11 * 1024 * 1024)
        mock_large = (io.BytesIO(large_bytes), "oversized.jpg")
        res = self.client.post(
            "/api/caption",
            data={"image": mock_large},
            content_type="multipart/form-data"
        )
        self.assertEqual(res.status_code, 413)
        data = res.get_json()
        self.assertFalse(data["success"])
        self.assertIn("exceeds maximum allowed limit", data["message"])

    def test_3_11_5_and_3_11_16_17_18_caption_generation_pipeline(self):
        """3.11.5, 3.11.16, 3.11.17, 3.11.18: Test caption generation and automatic result storage."""
        img_upload = (io.BytesIO(self.valid_image_bytes), "sky.jpg")
        response = self.client.post(
            "/api/caption",
            data={"image": img_upload, "style": "short", "count": "1"},
            content_type="multipart/form-data"
        )

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertIn("id", data)
        self.assertTrue(data["id"].startswith("cap_"))
        self.assertIn("caption", data)
        self.assertGreater(len(data["caption"]), 0)
        self.assertEqual(data["style"], "short")
        self.assertIn("confidence", data)
        self.assertIn("semantics", data)
        self.assertIn("quality_evaluation", data)
        self.assertIn("inference_time_ms", data)

        # Verify record was persisted in result_manager
        saved = result_manager.get_by_id(data["id"])
        self.assertIsNotNone(saved)
        self.assertEqual(saved["caption"], data["caption"])

        return data["id"]

    def test_3_11_6_and_3_11_8_caption_retrieval_by_id(self):
        """3.11.6 & 3.11.8: Test caption retrieval by ID via /api/caption/<id> and /api/history/<id>."""
        # 1. Generate a caption first
        img_upload = (io.BytesIO(self.valid_image_bytes), "test_retrieve.jpg")
        gen_res = self.client.post(
            "/api/caption",
            data={"image": img_upload, "style": "standard"},
            content_type="multipart/form-data"
        )
        result_id = gen_res.get_json()["id"]

        # 2. Retrieve via /api/caption/<id>
        res1 = self.client.get(f"/api/caption/{result_id}")
        self.assertEqual(res1.status_code, 200)
        d1 = res1.get_json()
        self.assertTrue(d1["success"])
        self.assertEqual(d1["id"], result_id)

        # 3. Retrieve via /api/history/<id>
        res2 = self.client.get(f"/api/history/{result_id}")
        self.assertEqual(res2.status_code, 200)
        d2 = res2.get_json()
        self.assertEqual(d2["id"], result_id)

        # 4. Non-existent ID returns 404
        res3 = self.client.get("/api/caption/non_existent_123")
        self.assertEqual(res3.status_code, 404)

    def test_3_11_7_caption_history_api(self):
        """3.11.7: Test caption history list endpoint /api/history."""
        # Seed 2 entries
        for i in range(2):
            img_upload = (io.BytesIO(self.valid_image_bytes), f"item_{i}.jpg")
            self.client.post(
                "/api/caption",
                data={"image": img_upload, "style": "standard"},
                content_type="multipart/form-data"
            )

        res = self.client.get("/api/history?limit=10&offset=0")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["total"], 2)
        self.assertEqual(len(data["items"]), 2)

    def test_3_11_9_caption_deletion_apis(self):
        """3.11.9: Test deleting individual history item and clearing all history."""
        # Generate 2 captions
        ids = []
        for i in range(2):
            img_upload = (io.BytesIO(self.valid_image_bytes), f"del_{i}.jpg")
            r = self.client.post(
                "/api/caption",
                data={"image": img_upload, "style": "standard"},
                content_type="multipart/form-data"
            )
            ids.append(r.get_json()["id"])

        self.assertEqual(result_manager.count(), 2)

        # 1. Delete single item
        del_res = self.client.delete(f"/api/history/{ids[0]}")
        self.assertEqual(del_res.status_code, 200)
        self.assertEqual(result_manager.count(), 1)

        # Deleting already deleted item returns 404
        del_res2 = self.client.delete(f"/api/history/{ids[0]}")
        self.assertEqual(del_res2.status_code, 404)

        # 2. Clear all remaining history
        clear_res = self.client.delete("/api/history")
        self.assertEqual(clear_res.status_code, 200)
        self.assertEqual(result_manager.count(), 0)

    def test_3_11_10_caption_regeneration_api(self):
        """3.11.10: Test caption regeneration API /api/caption/regenerate."""
        # Generate initial caption
        img_upload = (io.BytesIO(self.valid_image_bytes), "regen_source.jpg")
        initial_res = self.client.post(
            "/api/caption",
            data={"image": img_upload, "style": "standard"},
            content_type="multipart/form-data"
        )
        initial_data = initial_res.get_json()
        result_id = initial_data["id"]

        # Call /api/caption/regenerate with existing result_id
        regen_res = self.client.post(
            "/api/caption/regenerate",
            data=json.dumps({"result_id": result_id, "style": "creative"}),
            content_type="application/json"
        )
        self.assertEqual(regen_res.status_code, 200)
        regen_data = regen_res.get_json()
        self.assertTrue(regen_data["success"])
        self.assertTrue(regen_data["is_regenerated"])
        self.assertIn("caption", regen_data)

    def test_3_11_11_api_error_handlers(self):
        """3.11.11: Test global error handlers (404, 405)."""
        # Test 404 handler
        res_404 = self.client.get("/api/unknown-route-xyz")
        self.assertEqual(res_404.status_code, 404)
        d_404 = res_404.get_json()
        self.assertFalse(d_404["success"])
        self.assertEqual(d_404["status_code"], 404)

        # Test 405 Method Not Allowed handler
        res_405 = self.client.put("/api/health")
        self.assertEqual(res_405.status_code, 405)
        d_405 = res_405.get_json()
        self.assertFalse(d_405["success"])
        self.assertEqual(d_405["status_code"], 405)


if __name__ == "__main__":
    unittest.main()
