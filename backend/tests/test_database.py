import os
import sys
import unittest
from datetime import datetime
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import create_app
from app.database import db
from app.models import ImageMetadata, CaptionResult
from app.services.db_service import db_service
from app.config import Config


class TestDatabaseAndPersistence(unittest.TestCase):
    """
    Comprehensive Database Test Suite verifying Milestone 3.13:
    Schema Design, Models, Relationships, SQLAlchemy/MySQL config, CRUD,
    Timestamps, Cascade Deletion, Error Handling, and Persistence.
    """

    @classmethod
    def setUpClass(cls):
        os.environ["DATABASE_URL"] = "sqlite:///:memory:"
        cls.app = create_app()
        cls.app.config["TESTING"] = True
        cls.app_context = cls.app.app_context()
        cls.app_context.push()

    @classmethod
    def tearDownClass(cls):
        cls.app_context.pop()

    def setUp(self):
        """Recreate clean tables before each test."""
        db.create_all()
        db_service.clear_all_captions()

    def tearDown(self):
        db.session.remove()

    def test_3_13_1_schema_design_and_tables(self):
        """3.13.1: Verify tables exist in database metadata."""
        tables = db.metadata.tables.keys()
        self.assertIn("image_metadata", tables)
        self.assertIn("caption_result", tables)
        print("3.13.1 -> Database schema designed and verified: tables 'image_metadata' & 'caption_result' exist.")

    def test_3_13_2_and_3_13_8_image_metadata_model(self):
        """3.13.2 & 3.13.8: Create and store ImageMetadata model."""
        img = ImageMetadata(
            image_uuid="uuid-12345",
            filename="nature_photo.jpg",
            original_name="nature.jpg",
            width=1920,
            height=1080,
            format="JPEG",
            size_kb=420.5
        )
        db.session.add(img)
        db.session.commit()

        retrieved = ImageMetadata.query.filter_by(image_uuid="uuid-12345").first()
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.filename, "nature_photo.jpg")
        self.assertEqual(retrieved.width, 1920)
        self.assertEqual(retrieved.height, 1080)
        self.assertIsInstance(retrieved.created_at, datetime)
        print("3.13.2 & 3.13.8 -> ImageMetadata model created and stored successfully.")

    def test_3_13_3_and_3_13_9_and_3_13_10_caption_result_model_and_timestamp(self):
        """3.13.3, 3.13.9, 3.13.10: Create CaptionResult, store caption, verify timestamp."""
        img = ImageMetadata(
            image_uuid="uuid-caption-test",
            filename="city.jpg",
            original_name="city.jpg",
            width=800,
            height=600,
            format="JPEG",
            size_kb=150.0
        )
        db.session.add(img)
        db.session.flush()

        caption = CaptionResult(
            caption_id="cap_test_001",
            image_id=img.id,
            caption_text="A vibrant modern city skyline illuminated at night.",
            style="detailed",
            confidence_score=94.5,
            confidence_level="High",
            word_count=8,
            char_count=52,
            inference_time_ms=1950.0,
            semantics_json=json.dumps({"objects": ["Buildings", "Lights"], "scene": "City"}),
            quality_score=92.0,
            quality_grade="High Quality",
            is_regenerated=False
        )
        db.session.add(caption)
        db.session.commit()

        saved_cap = CaptionResult.query.filter_by(caption_id="cap_test_001").first()
        self.assertIsNotNone(saved_cap)
        self.assertEqual(saved_cap.style, "detailed")
        self.assertEqual(saved_cap.confidence_score, 94.5)
        self.assertIsInstance(saved_cap.created_at, datetime)  # 3.13.10
        print("3.13.3, 3.13.9, 3.13.10 -> CaptionResult and timestamp stored successfully.")

    def test_3_13_4_and_3_13_14_database_relationships_and_cascade(self):
        """3.13.4 & 3.13.14: Test 1-to-many relationship and cascading deletion."""
        img = ImageMetadata(
            image_uuid="uuid-rel-test",
            filename="multicaption.jpg",
            width=500,
            height=500
        )
        db.session.add(img)
        db.session.flush()

        # Add 3 captions to the same image (1-to-many)
        c1 = CaptionResult(caption_id="cap_var_1", image_id=img.id, caption_text="Caption 1")
        c2 = CaptionResult(caption_id="cap_var_2", image_id=img.id, caption_text="Caption 2")
        c3 = CaptionResult(caption_id="cap_var_3", image_id=img.id, caption_text="Caption 3")
        db.session.add_all([c1, c2, c3])
        db.session.commit()

        # Verify 1-to-many relationship
        reloaded_img = ImageMetadata.query.filter_by(image_uuid="uuid-rel-test").first()
        self.assertEqual(len(reloaded_img.captions), 3)

        # Verify backref
        self.assertEqual(c1.image.image_uuid, "uuid-rel-test")

        # Test Cascade Delete: Deleting parent image deletes all 3 child captions
        db.session.delete(reloaded_img)
        db.session.commit()

        remaining_caps = CaptionResult.query.filter(CaptionResult.caption_id.in_(["cap_var_1", "cap_var_2", "cap_var_3"])).count()
        self.assertEqual(remaining_caps, 0, "Cascade delete should have removed all 3 child captions.")
        print("3.13.4 & 3.13.14 -> 1-to-many relationship & cascade delete verified.")

    def test_3_13_5_and_3_13_6_sqlalchemy_and_mysql_configuration(self):
        """3.13.5 & 3.13.6: Verify SQLAlchemy and MySQL configuration attributes."""
        self.assertIsNotNone(Config.SQLALCHEMY_DATABASE_URI)
        self.assertFalse(Config.SQLALCHEMY_TRACK_MODIFICATIONS)
        self.assertIn("pool_recycle", Config.SQLALCHEMY_ENGINE_OPTIONS)
        self.assertEqual(Config.MYSQL_PORT, "3306")
        self.assertTrue(hasattr(Config, "MYSQL_HOST"))
        self.assertTrue(hasattr(Config, "MYSQL_DB"))
        print("3.13.5 & 3.13.6 -> SQLAlchemy & MySQL parameters configured and verified.")

    def test_3_13_7_crud_operations_via_service(self):
        """3.13.7: Test full CRUD operations via DatabaseService."""
        # 1. Create (Insert)
        img_info = {
            "filename": "crud_test.jpg",
            "original_name": "crud.jpg",
            "width": 640,
            "height": 480,
            "format": "JPEG",
            "size_kb": 120.0
        }
        res = db_service.save_caption_with_image(
            image_info=img_info,
            caption_text="A playful puppy rolling on the rug.",
            captions_list=[{"id": 1, "caption": "A playful puppy rolling on the rug."}],
            style="creative",
            confidence={"score": 93.0, "level": "High"},
            inference_time_ms=1650.0
        )
        self.assertIn("id", res)
        caption_id = res["id"]

        # 2. Read (Query single & all)
        single = db_service.get_caption_by_id(caption_id)
        self.assertIsNotNone(single)
        self.assertEqual(single["caption"], "A playful puppy rolling on the rug.")

        all_res = db_service.get_all_captions(limit=10, offset=0)
        self.assertEqual(all_res["total"], 1)

        # 3. Delete
        del_success = db_service.delete_caption_by_id(caption_id)
        self.assertTrue(del_success)
        self.assertEqual(db_service.get_count(), 0)
        print("3.13.7 -> CRUD operations verified successfully.")

    def test_3_13_11_retrieve_caption_history_pagination(self):
        """3.13.11: Retrieve caption history with pagination."""
        for i in range(5):
            db_service.save_caption_with_image(
                image_info={"filename": f"page_{i}.jpg"},
                caption_text=f"Caption number {i}",
                captions_list=[]
            )

        page1 = db_service.get_all_captions(limit=3, offset=0)
        self.assertEqual(page1["total"], 5)
        self.assertEqual(len(page1["items"]), 3)

        page2 = db_service.get_all_captions(limit=3, offset=3)
        self.assertEqual(len(page2["items"]), 2)
        print("3.13.11 -> History retrieval pagination verified.")

    def test_3_13_12_delete_records_and_clear_all(self):
        """3.13.12: Delete specific caption records and clear all."""
        r1 = db_service.save_caption_with_image(image_info={"filename": "a.jpg"}, caption_text="Cap A", captions_list=[])
        r2 = db_service.save_caption_with_image(image_info={"filename": "b.jpg"}, caption_text="Cap B", captions_list=[])
        self.assertEqual(db_service.get_count(), 2)

        db_service.delete_caption_by_id(r1["id"])
        self.assertEqual(db_service.get_count(), 1)

        cleared = db_service.clear_all_captions()
        self.assertEqual(cleared, 1)
        self.assertEqual(db_service.get_count(), 0)
        print("3.13.12 -> Delete caption records and clear all verified.")

    def test_3_13_13_handle_database_errors_and_rollback(self):
        """3.13.13: Test database error handling and session rollback."""
        # Add initial image
        img1 = ImageMetadata(image_uuid="dup-uuid", filename="img1.jpg")
        db.session.add(img1)
        db.session.commit()

        # Attempting to save another ImageMetadata with duplicate unique image_uuid
        with self.assertRaises(Exception):
            img2 = ImageMetadata(image_uuid="dup-uuid", filename="img2.jpg")
            db.session.add(img2)
            db.session.commit()

        # Ensure rollback occurs and session is clean and recovered
        db.session.rollback()
        self.assertEqual(ImageMetadata.query.count(), 1)
        print("3.13.13 -> Error handling and session rollback verified.")

    def test_3_13_15_persistent_storage_verification(self):
        """3.13.15: Verify data survives session close and reconnect."""
        # Save record
        res = db_service.save_caption_with_image(
            image_info={"filename": "persistent.jpg", "width": 800, "height": 600},
            caption_text="Persistent mountain landscape across session boundaries.",
            captions_list=[]
        )
        cap_id = res["id"]

        # Completely close and remove session
        db.session.remove()

        # Re-query in a brand new session
        record = db_service.get_caption_by_id(cap_id)
        self.assertIsNotNone(record)
        self.assertEqual(record["caption"], "Persistent mountain landscape across session boundaries.")
        print("3.13.15 -> Persistent storage verified: data survives session closure!")


if __name__ == "__main__":
    unittest.main()
