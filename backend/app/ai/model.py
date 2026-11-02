import os
import threading
import torch
import psutil
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration


# Hugging Face BLIP model
MODEL_NAME = "Salesforce/blip-image-captioning-base"

_processor = None
_model = None
_device = None
_model_lock = threading.RLock()
_is_warmed_up = False


def get_model_lock():
    """Return the global model threading lock for safe concurrency."""
    return _model_lock


def get_memory_usage_mb() -> float:
    """Return current process resident set memory (RSS) in megabytes."""
    try:
        proc = psutil.Process(os.getpid())
        return round(proc.memory_info().rss / (1024 * 1024), 1)
    except Exception:
        return 0.0


def get_device():
    """Return torch device (cuda if available, else cpu)."""
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Optimize CPU threads for inference if running on CPU
        if _device.type == "cpu":
            num_cores = max(1, os.cpu_count() or 4)
            torch.set_num_threads(min(num_cores, 4))
    return _device


def is_model_loaded():
    """Check if the model is currently loaded in memory."""
    global _processor, _model
    return _processor is not None and _model is not None


def load_model():
    """
    Load the BLIP processor and image captioning model.
    The model is loaded only once (singleton pattern) and reused across requests.
    Thread-safe initialization.
    """
    global _processor, _model

    if _processor is None or _model is None:
        with _model_lock:
            if _processor is None or _model is None:
                device = get_device()
                print(f"[BLIP] Loading model '{MODEL_NAME}' on {device}...")

                try:
                    _processor = BlipProcessor.from_pretrained(MODEL_NAME)
                    _model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)
                    _model.to(device)
                    _model.eval()  # Set to evaluation mode for inference
                    print(f"[BLIP] Model loaded successfully on {device}!")
                except Exception as e:
                    print(f"[BLIP Error] Failed to load model: {e}")
                    raise e

    return _processor, _model


def warmup_model():
    """
    Pre-run a single forward pass on dummy input to pre-compile PyTorch graphs
    and pre-allocate memory buffers, eliminating first-request latency.
    """
    global _is_warmed_up
    if _is_warmed_up:
        return True

    processor, model = load_model()
    device = get_device()

    with _model_lock:
        if _is_warmed_up:
            return True

        print("[BLIP] Warming up model with dummy pass...")
        try:
            dummy_img = Image.new("RGB", (384, 384), color=(128, 128, 128))
            inputs = processor(images=dummy_img, return_tensors="pt").to(device)
            with torch.inference_mode():
                _ = model.generate(**inputs, max_new_tokens=5, num_beams=1)
            _is_warmed_up = True
            print(f"[BLIP] Warmup complete! RAM in use: {get_memory_usage_mb()} MB")
            return True
        except Exception as e:
            print(f"[BLIP] Warmup warning (non-fatal): {e}")
            return False
