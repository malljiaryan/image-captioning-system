from PIL import Image
import numpy as np
import io


MAX_IMAGE_DIM = 1200  # Max dimension cap for performance and memory guard


def load_image_for_model(image_source):
    """
    Safely open and validate an image for model processing.
    Includes large image downscaling protection to prevent memory bloat.
    Accepts:
        - file path (str)
        - file-like object / bytes
        - PIL Image instance
    Returns:
        PIL.Image in RGB mode
    """
    if isinstance(image_source, Image.Image):
        image = image_source
    elif isinstance(image_source, (bytes, bytearray)):
        image = Image.open(io.BytesIO(image_source))
    else:
        image = Image.open(image_source)

    # Convert to RGB (handles RGBA, grayscale, CMYK, etc.)
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Basic dimensional validation
    width, height = image.size
    if width < 10 or height < 10:
        raise ValueError(f"Image dimensions ({width}x{height}) are too small for processing.")

    # High-resolution downscale protection (e.g. 4K/8K photos)
    if width > MAX_IMAGE_DIM or height > MAX_IMAGE_DIM:
        # Use fast Bicubic resampling
        image.thumbnail((MAX_IMAGE_DIM, MAX_IMAGE_DIM), Image.Resampling.BICUBIC)

    return image


def preprocess_image(image_path, target_size=(224, 224)):
    """
    Load and preprocess an image for educational/tensor demonstration.

    Steps:
    1. Open image
    2. Convert to RGB
    3. Resize
    4. Convert to NumPy array
    5. Normalize pixel values
    """
    # Open image
    image = Image.open(image_path)

    # Convert image to RGB
    image = image.convert("RGB")

    # Resize image
    image = image.resize(target_size)

    # Convert to NumPy array
    image_array = np.array(image, dtype=np.float32)

    # Normalize pixel values from [0, 255] to [0, 1]
    image_array = image_array / 255.0

    return image_array