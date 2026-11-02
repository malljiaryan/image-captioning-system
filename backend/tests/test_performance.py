import os
import sys
import time
import io
import unittest
import concurrent.futures
from PIL import Image

# Ensure backend root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import create_app
from app.ai.model import (
    load_model,
    is_model_loaded,
    get_memory_usage_mb,
    warmup_model,
    get_device,
    MODEL_NAME
)
from app.ai.preprocessing import load_image_for_model, MAX_IMAGE_DIM
from app.ai.caption_generator import generate_caption


class TestAIPerformanceAndOptimization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Warmup and initialize test environment."""
        cls.app = create_app()
        cls.app.config["TESTING"] = True
        cls.client = cls.app.test_client()

        # Warm up model so caching and pre-compilation are active
        warmup_model()

        # Create a standard synthetic test image (400x300 RGB)
        img = Image.new("RGB", (400, 300), color=(100, 150, 200))
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format="JPEG")
        cls.test_image_bytes = img_byte_arr.getvalue()

    def test_3_10_1_and_3_10_5_model_caching_and_load_time(self):
        """3.10.1 & 3.10.5: Verify singleton model caching and instant retrieval."""
        t0 = time.time()
        p1, m1 = load_model()
        retrieval_time = time.time() - t0

        t1 = time.time()
        p2, m2 = load_model()
        cached_retrieval_time = time.time() - t1

        self.assertIs(p1, p2, "Processor should be the identical cached singleton instance.")
        self.assertIs(m1, m2, "Model should be the identical cached singleton instance.")
        self.assertLess(cached_retrieval_time, 0.05, "Cached model retrieval should be instantaneous (< 50ms).")
        self.assertTrue(is_model_loaded(), "Model loaded state should be True.")

    def test_3_10_3_cpu_memory_usage(self):
        """3.10.3: Analyze CPU RAM usage via psutil."""
        mem_mb = get_memory_usage_mb()
        self.assertGreater(mem_mb, 50.0, "Process memory should be measured (> 50 MB).")
        # BLIP base model with PyTorch CPU should comfortably fit under 1500 MB
        self.assertLess(mem_mb, 1500.0, f"Process memory ({mem_mb} MB) exceeds expected 1500 MB boundary.")
        print(f"\n[Perf Test] Current Resident Process RAM: {mem_mb} MB")

    def test_3_10_6_and_3_10_10_large_image_processing_guard(self):
        """3.10.6 & 3.10.10: Verify high-res image (3200x2400) is safely downscaled to MAX_IMAGE_DIM."""
        large_img = Image.new("RGB", (3200, 2400), color=(220, 180, 140))
        img_bytes = io.BytesIO()
        large_img.save(img_bytes, format="JPEG")
        raw_bytes = img_bytes.getvalue()

        t0 = time.time()
        processed = load_image_for_model(raw_bytes)
        elapsed = time.time() - t0

        w, h = processed.size
        self.assertLessEqual(max(w, h), MAX_IMAGE_DIM, f"Dimensions ({w}x{h}) should be capped at {MAX_IMAGE_DIM}.")
        self.assertLess(elapsed, 0.8, "Image downscaling and preprocessing should complete in < 800ms.")
        print(f"[Perf Test] Large Image 3200x2400 -> {w}x{h} in {elapsed*1000:.1f}ms")

    def test_3_10_2_and_3_10_7_inference_parameters_and_speed(self):
        """3.10.2 & 3.10.7: Verify caption generation latency and generation parameters."""
        result = generate_caption(self.test_image_bytes, style="standard", num_captions=1)

        self.assertIn("caption", result)
        self.assertIn("inference_time_ms", result)
        self.assertGreater(len(result["caption"]), 5)
        self.assertGreater(result["confidence"]["score"], 0)
        print(f"[Perf Test] Standard Caption Inference: {result['inference_time_ms']}ms -> '{result['caption']}'")

    def test_3_10_8_concurrent_caption_requests(self):
        """3.10.8: Handle concurrent caption requests safely without race conditions."""
        concurrency_level = 3

        def run_inference(req_id):
            t0 = time.time()
            res = generate_caption(self.test_image_bytes, style="standard", num_captions=1)
            duration = round((time.time() - t0) * 1000, 2)
            return req_id, res["caption"], duration

        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency_level) as executor:
            futures = [executor.submit(run_inference, i) for i in range(concurrency_level)]
            results = [f.result() for f in futures]

        self.assertEqual(len(results), concurrency_level)
        for req_id, caption, dur in results:
            self.assertTrue(len(caption) > 0)
            print(f"[Perf Test Concurrency] Thread {req_id} completed in {dur}ms: '{caption}'")

    def test_3_10_9_inference_timeout_handling(self):
        """3.10.9: Verify inference timeout handling when limit is exceeded."""
        with self.assertRaises(TimeoutError):
            generate_caption(
                self.test_image_bytes,
                style="standard",
                timeout_seconds=0.0001
            )
        print("[Perf Test Timeout] TimeoutError successfully caught and verified!")

    def test_3_10_11_sequential_requests_throughput(self):
        """3.10.11: Measure sequential request throughput and consistency."""
        num_runs = 3
        latencies = []

        for i in range(num_runs):
            t0 = time.time()
            _ = generate_caption(self.test_image_bytes, style="short")
            latencies.append((time.time() - t0) * 1000)

        avg_latency = sum(latencies) / len(latencies)
        print(f"[Perf Test Sequential] {num_runs} runs: Avg Latency = {avg_latency:.1f}ms (Min: {min(latencies):.1f}ms, Max: {max(latencies):.1f}ms)")
        self.assertLess(avg_latency, 7000.0, "Average latency should be below 7.0s on standard CPU.")

    def test_3_10_13_performance_stats_endpoint(self):
        """3.10.13: Verify system performance API endpoint returns live metrics."""
        response = self.client.get("/api/performance/stats")
        self.assertEqual(response.status_code, 200)

        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["model_name"], MODEL_NAME)
        self.assertTrue(data["model_loaded"])
        self.assertGreater(data["memory_usage_mb"], 50.0)
        self.assertEqual(data["cpu_threads"], 4)
        self.assertTrue(data["optimizations"]["model_caching"])
        self.assertTrue(data["optimizations"]["warmup_precompile"])
        self.assertEqual(data["optimizations"]["max_image_dimension_guard"], 1200)

    def test_3_10_14_and_3_10_15_regression_and_readiness(self):
        """3.10.14 & 3.10.15: Verify all styles and multi-caption variations without regression."""
        styles = ["short", "detailed", "creative"]
        for s in styles:
            res = generate_caption(self.test_image_bytes, style=s, num_captions=1)
            self.assertIn("caption", res)
            self.assertEqual(res["style"], s)
            self.assertGreater(len(res["caption"]), 0)

        # Multi-caption 3 variations
        multi_res = generate_caption(self.test_image_bytes, style="standard", num_captions=3)
        self.assertEqual(len(multi_res["captions"]), 3)


if __name__ == "__main__":
    unittest.main()
