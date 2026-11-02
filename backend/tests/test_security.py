import os
import sys
import io
import unittest
from unittest.mock import patch
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import create_app
from app.utils.validators import sanitize_filename, verify_magic_bytes, validate_image_file
from app.config import Config
from app.services.db_service import db_service


class TestSecurityAndHardening(unittest.TestCase):
    """
    Automated Security Verification Suite for Milestone 3.14:
    - Path traversal sanitization
    - Dangerous double extension blocking
    - Magic byte binary MIME verification
    - Upload quota enforcement (10MB limit)
    - SQL injection resistance
    - HTTP security headers
    """

    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.app.config["TESTING"] = True
        cls.client = cls.app.test_client()

        # Valid JPEG test image
        img = Image.new("RGB", (200, 200), color=(80, 120, 160))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        cls.valid_jpeg_bytes = buf.getvalue()

    def test_3_14_1_and_3_14_5_path_traversal_and_null_bytes(self):
        """3.14.1 & 3.14.5: Verify path traversal and null byte sanitization."""
        malicious_names = [
            "../../etc/passwd.jpg",
            "..\\..\\windows\\system32\\cmd.exe.jpg",
            "/var/www/shell.jpg",
            "image.jpg\0.exe",
            "normal_image.jpg"
        ]

        sanitized_0 = sanitize_filename(malicious_names[0])
        self.assertTrue(sanitized_0 is None or ("/" not in sanitized_0 and ".." not in sanitized_0))

        sanitized_1 = sanitize_filename(malicious_names[1])
        self.assertTrue(sanitized_1 is None or ("\\" not in sanitized_1 and ".." not in sanitized_1))

        self.assertEqual(sanitize_filename(malicious_names[4]), "normal_image.jpg")

        # Null byte must be rejected completely
        self.assertIsNone(sanitize_filename("image.jpg\0.exe"))
        print("3.14.1 & 3.14.5 -> Path traversal & null-byte injection prevention verified.")

    def test_3_14_2_dangerous_double_extensions(self):
        """3.14.2: Verify blocking of executable double extensions."""
        bad_filenames = [
            "exploit.php.jpg",
            "shell.exe.png",
            "backdoor.py.webp",
            "script.js.jpg",
            "page.html.png"
        ]
        for name in bad_filenames:
            self.assertIsNone(sanitize_filename(name), f"Dangerous filename '{name}' should be rejected.")
        print("3.14.2 -> Dangerous double-extensions blocked successfully.")

    def test_3_14_3_mime_spoofing_magic_bytes(self):
        """3.14.3: Verify binary magic byte verification prevents MIME spoofing."""
        fake_jpeg = b"<?php echo 'malicious script'; ?>"  # Text file named .jpg
        self.assertFalse(verify_magic_bytes(fake_jpeg, "jpg"))

        # Real JPEG bytes should pass
        self.assertTrue(verify_magic_bytes(self.valid_jpeg_bytes, "jpg"))
        print("3.14.3 -> Binary magic byte MIME validation verified.")

    def test_3_14_4_upload_size_limit_enforcement(self):
        """3.14.4: Verify upload quota limit enforcement (> 10MB rejected)."""
        huge_bytes = b"0" * (11 * 1024 * 1024)
        mock_upload = (io.BytesIO(huge_bytes), "huge.jpg")

        res = self.client.post(
            "/api/caption",
            data={"image": mock_upload},
            content_type="multipart/form-data"
        )
        self.assertEqual(res.status_code, 413)
        self.assertFalse(res.get_json()["success"])
        print("3.14.4 -> Upload size limit (10 MB quota) verified.")

    def test_3_14_10_and_3_14_15_security_headers(self):
        """3.14.10 & 3.14.15: Verify production security response headers."""
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.headers.get("X-Content-Type-Options"), "nosniff")
        self.assertEqual(res.headers.get("X-Frame-Options"), "SAMEORIGIN")
        self.assertEqual(res.headers.get("X-XSS-Protection"), "1; mode=block")
        self.assertEqual(res.headers.get("Access-Control-Allow-Origin"), "*")
        print("3.14.10 & 3.14.15 -> Production security headers & CORS verified.")

    def test_3_14_14_sql_injection_resistance(self):
        """3.14.14: Verify parameterized SQL queries resist injection attacks."""
        sql_injections = [
            "' OR '1'='1",
            "'; DROP TABLE caption_result; --",
            "1 UNION SELECT * FROM image_metadata"
        ]
        for payload in sql_injections:
            # Querying with payload must not raise syntax error or dump unintended rows
            res = self.client.get(f"/api/caption/{payload}")
            self.assertEqual(res.status_code, 404, f"SQL injection payload '{payload}' should safely return 404.")
        print("3.14.14 -> SQL injection resistance verified via ORM parameterization.")


if __name__ == "__main__":
    unittest.main()
