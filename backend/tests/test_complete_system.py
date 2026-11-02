import os
import sys
import io
import unittest
import json
from unittest.mock import patch
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import create_app
from app.services.db_service import db_service
from app.ai.preprocessing import load_image_for_model, MAX_IMAGE_DIM
from app.ai.evaluator import evaluate_caption


class TestCompleteSystemAndAcceptance(unittest.TestCase):
    """
    End-to-End System Testing & Project Acceptance (Milestones 3.15 & 3.16)
    Verifies all functional layers: Upload, Preprocessing, Inference, Multi-Format,
    History, Database, Security, Error Resilience, and Quality Evaluation.
    """

    @classmethod
    def setUpClass(cls):
        os.environ["DATABASE_URL"] = "sqlite:///:memory:"
        cls.app = create_app()
        cls.app.config["TESTING"] = True
        cls.app_context = cls.app.app_context()
        cls.app_context.push()
        cls.client = cls.app.test_client()

        # Generate sample images in 3 formats: JPEG, PNG, WEBP
        cls.test_images = {}

        # 1. JPEG
        j_img = Image.new("RGB", (400, 300), color=(100, 150, 200))
        j_buf = io.BytesIO()
        j_img.save(j_buf, format="JPEG")
        cls.test_images["jpg"] = j_buf.getvalue()

        # 2. PNG
        p_img = Image.new("RGB", (300, 300), color=(150, 200, 100))
        p_buf = io.BytesIO()
        p_img.save(p_buf, format="PNG")
        cls.test_images["png"] = p_buf.getvalue()

        # 3. WEBP
        w_img = Image.new("RGB", (350, 250), color=(200, 120, 120))
        w_buf = io.BytesIO()
        w_img.save(w_buf, format="WEBP")
        cls.test_images["webp"] = w_buf.getvalue()

    @classmethod
    def tearDownClass(cls):
        cls.app_context.pop()

    def setUp(self):
        db_service.clear_all_captions()

    def test_3_15_3_image_preprocessing_and_resampling(self):
        """3.15.3: Test preprocessing pipeline, color space conversion, and size guard."""
        # Ultra large image (2800 x 2100)
        large_img = Image.new("RGB", (2800, 2100), color=(50, 100, 150))
        buf = io.BytesIO()
        large_img.save(buf, format="JPEG")

        processed = load_image_for_model(buf.getvalue())
        w, h = processed.size
        self.assertLessEqual(max(w, h), MAX_IMAGE_DIM)
        self.assertEqual(processed.mode, "RGB")
        print(f"3.15.3 -> Image preprocessing verified: {2800}x{2100} safely capped to {w}x{h}.")

    @patch("app.routes.analyze_image_semantics")
    @patch("app.routes.generate_caption")
    def test_3_15_11_multiple_image_formats_jpg_png_webp(self, mock_gen, mock_sem):
        """3.15.11: Test system with multiple formats (JPEG, PNG, WEBP)."""
        mock_gen.return_value = {
            "caption": "A scenic test photograph.",
            "captions": [{"id": 1, "caption": "A scenic test photograph.", "confidence": {"score": 92.0, "level": "High"}, "text_stats": {"word_count": 4, "char_count": 26}}],
            "style": "standard",
            "confidence": {"score": 92.0, "level": "High"},
            "text_stats": {"word_count": 4, "char_count": 26},
            "inference_time_ms": 210.0,
            "model_metadata": {"model": "Salesforce/blip-image-captioning-base"}
        }
        mock_sem.return_value = {"objects": ["Subject"], "scene": "Studio"}

        for ext, img_bytes in self.test_images.items():
            upload_tuple = (io.BytesIO(img_bytes), f"test_format.{ext}")
            res = self.client.post(
                "/api/caption",
                data={"image": upload_tuple, "style": "standard"},
                content_type="multipart/form-data"
            )
            self.assertEqual(res.status_code, 200, f"Format .{ext} should be successfully processed.")
            data = res.get_json()
            self.assertTrue(data["success"])
            self.assertEqual(data["image_metadata"]["format"].lower(), ext.replace("jpg", "jpeg"))

        self.assertEqual(db_service.get_count(), 3)
        print("3.15.11 -> Multiple image formats (JPG, PNG, WEBP) processed successfully.")

    @patch("app.routes.analyze_image_semantics")
    @patch("app.routes.generate_caption")
    def test_3_15_4_and_3_15_5_caption_styles_and_variations(self, mock_gen, mock_sem):
        """3.15.4 & 3.15.5: Test style parameterization and 3 candidate variations."""
        mock_sem.return_value = {"objects": ["Mountain"], "scene": "Outdoors"}
        mock_gen.return_value = {
            "caption": "A detailed panoramic vista of snow-capped mountains.",
            "captions": [
                {"id": 1, "caption": "A detailed panoramic vista of snow-capped mountains.", "confidence": {"score": 96.0, "level": "High"}, "text_stats": {"word_count": 8, "char_count": 52}},
                {"id": 2, "caption": "Snow-covered mountain peaks reaching the sky.", "confidence": {"score": 93.0, "level": "High"}, "text_stats": {"word_count": 6, "char_count": 45}},
                {"id": 3, "caption": "Breathtaking mountain ridges in early morning light.", "confidence": {"score": 91.0, "level": "High"}, "text_stats": {"word_count": 7, "char_count": 53}}
            ],
            "style": "detailed",
            "confidence": {"score": 96.0, "level": "High"},
            "text_stats": {"word_count": 8, "char_count": 52},
            "inference_time_ms": 320.0,
            "model_metadata": {"model": "Salesforce/blip-image-captioning-base"}
        }

        upload_tuple = (io.BytesIO(self.test_images["jpg"]), "mountains.jpg")
        res = self.client.post(
            "/api/caption",
            data={"image": upload_tuple, "style": "detailed", "count": "3"},
            content_type="multipart/form-data"
        )
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(len(data["captions"]), 3)
        self.assertEqual(data["style"], "detailed")
        print("3.15.4 & 3.15.5 -> Multi-variation generation and detailed style verified.")

    def test_3_15_6_and_3_15_7_caption_history_and_persistence(self):
        """3.15.6 & 3.15.7: Test caption history lifecycle in database."""
        # Insert 3 records
        for i in range(3):
            db_service.save_caption_with_image(
                image_info={"filename": f"sample_{i}.jpg", "width": 640, "height": 480},
                caption_text=f"Historical caption entry number {i}.",
                captions_list=[]
            )

        # Retrieve history
        hist = self.client.get("/api/history?limit=10").get_json()["data"]
        self.assertEqual(hist["total"], 3)
        self.assertEqual(len(hist["items"]), 3)

        # Delete 1 record
        target_id = hist["items"][0]["id"]
        del_res = self.client.delete(f"/api/history/{target_id}")
        self.assertEqual(del_res.status_code, 200)

        # Verify updated count
        after = self.client.get("/api/history").get_json()["data"]
        self.assertEqual(after["total"], 2)
        print("3.15.6 & 3.15.7 -> Caption history and database persistence verified.")

    def test_3_15_8_and_3_15_9_error_and_invalid_file_handling(self):
        """3.15.8 & 3.15.9: Test error responses on invalid requests."""
        # 1. No image uploaded
        r1 = self.client.post("/api/caption", data={})
        self.assertEqual(r1.status_code, 400)

        # 2. Corrupted file content
        corrupted = (io.BytesIO(b"NOT_A_REAL_IMAGE"), "corrupt.jpg")
        r2 = self.client.post("/api/caption", data={"image": corrupted}, content_type="multipart/form-data")
        self.assertEqual(r2.status_code, 400)

        # 3. Disallowed extension (.exe)
        bad_ext = (io.BytesIO(b"MZ\x90\x00"), "virus.exe")
        r3 = self.client.post("/api/caption", data={"image": bad_ext}, content_type="multipart/form-data")
        self.assertEqual(r3.status_code, 400)
        print("3.15.8 & 3.15.9 -> Error handling & invalid file rejection verified.")

    def test_3_16_5_caption_quality_evaluation_benchmark(self):
        """3.16.5: Verify NLP evaluation benchmark (BLEU, ROUGE, METEOR, Similarity)."""
        candidate = "A black dog is running on the green grass."
        reference = "A black dog is running across the green lawn."

        eval_res = evaluate_caption(candidate, reference=reference)
        self.assertIn("composite_score", eval_res)
        self.assertIn("bleu", eval_res)
        self.assertIn("rouge", eval_res)
        self.assertIn("meteor", eval_res)
        self.assertIn("semantic_similarity", eval_res)
        self.assertGreater(eval_res["composite_score"], 40.0)
        print(f"3.16.5 -> NLP Quality Benchmark verified: Score = {eval_res['composite_score']}/100, Grade = '{eval_res['grade']}'")


if __name__ == "__main__":
    unittest.main()
