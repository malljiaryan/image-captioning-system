import os
import sys

# Ensure backend root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.ai.model import load_model, get_device, is_model_loaded

print("--- Step 3.6.7: Verify BLIP Model Loading ---")
print(f"Device detected: {get_device()}")
print(f"Initial loaded state: {is_model_loaded()}")

processor, model = load_model()

print("\nBLIP model loaded successfully!")
print("Processor:", type(processor).__name__)
print("Model:", type(model).__name__)
print(f"Loaded state after call: {is_model_loaded()}")

# Verify singleton (identical object reference)
processor2, model2 = load_model()
assert processor is processor2, "Processor singleton check failed!"
assert model is model2, "Model singleton check failed!"
print("Singleton pattern verified: same instances reused.\n")