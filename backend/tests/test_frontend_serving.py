import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["TESTING"] = "true" 

from app import create_app


class TestFrontendWebServing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = create_app()
        cls.app.config["TESTING"] = True
        cls.client = cls.app.test_client()

    def test_home_page_serves_html(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn("text/html", res.content_type)
        html = res.get_data(as_text=True)
        self.assertIn("VisionCaption AI", html)
        self.assertIn("<!DOCTYPE html>", html)

    def test_caption_page_serves_html(self):
        res = self.client.get("/caption")
        self.assertEqual(res.status_code, 200)
        self.assertIn("text/html", res.content_type)
        html = res.get_data(as_text=True)
        self.assertIn("Generate a Caption", html)

    def test_history_page_serves_html(self):
        res = self.client.get("/history")
        self.assertEqual(res.status_code, 200)
        self.assertIn("text/html", res.content_type)
        html = res.get_data(as_text=True)
        self.assertIn("Your Caption History", html)

    def test_about_page_serves_html(self):
        res = self.client.get("/about")
        self.assertEqual(res.status_code, 200)
        self.assertIn("text/html", res.content_type)
        html = res.get_data(as_text=True)
        self.assertIn("How It Works", html)

    def test_api_discovery_endpoint(self):
        res = self.client.get("/api")
        self.assertEqual(res.status_code, 200)
        self.assertIn("application/json", res.content_type)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertIn("endpoints", data["data"])

    def test_static_css_and_js_served(self):
        res_css = self.client.get("/css/style.css")
        self.assertEqual(res_css.status_code, 200)
        self.assertIn("text/css", res_css.content_type)

        res_resp = self.client.get("/css/responsive.css")
        self.assertEqual(res_resp.status_code, 200)
        self.assertIn("text/css", res_resp.content_type)

        res_js = self.client.get("/js/app.js")
        self.assertEqual(res_js.status_code, 200)
        self.assertIn("showToast", res_js.get_data(as_text=True))


if __name__ == "__main__":
    unittest.main()
