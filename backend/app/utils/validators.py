import os
import re
from werkzeug.utils import secure_filename
from PIL import Image
import io
from typing import Tuple, Optional, Dict, Any

# Magic Byte Signatures for Image Validation (3.14.3)
MAGIC_SIGNATURES = {
    "jpg": [b"\xFF\xD8\xFF"],
    "jpeg": [b"\xFF\xD8\xFF"],
    "png": [b"\x89PNG\r\n\x1a\n"],
    "webp": [b"RIFF"]
}

# Dangerous executable extensions blocked even in double extensions (3.14.1, 3.14.2)
DANGEROUS_EXTENSIONS = {
    "php", "phtml", "py", "pyc", "exe", "bat", "cmd", "sh", "bash",
    "vbs", "ps1", "js", "mjs", "html", "htm", "svg", "jsp", "asp", "aspx"
}


def sanitize_filename(raw_filename: str) -> Optional[str]:
    """
    Sanitize and protect filename from path traversal and injection attacks (3.14.5).
    Rejects filenames containing path separators, null bytes, or dangerous sub-extensions.
    """
    if not raw_filename or "\0" in raw_filename:
        return None

    # Strip directory components
    clean_name = os.path.basename(raw_filename)
    clean_name = secure_filename(clean_name)

    if not clean_name or "." not in clean_name:
        return None

    # Check against dangerous double extensions (e.g., shell.php.jpg)
    parts = clean_name.lower().split(".")
    for part in parts[:-1]:
        if part in DANGEROUS_EXTENSIONS:
            return None

    return clean_name


def verify_magic_bytes(raw_bytes: bytes, extension: str) -> bool:
    """
    Validate binary file signature (magic bytes) against declared extension (3.14.3).
    Prevents MIME-type spoofing (e.g. text/php script disguised as .jpg).
    """
    ext = extension.lower().strip()
    signatures = MAGIC_SIGNATURES.get(ext)
    if not signatures:
        return False

    for sig in signatures:
        if raw_bytes.startswith(sig):
            if ext == "webp":
                # WEBP format requires 'WEBP' identifier at bytes 8-12
                return len(raw_bytes) >= 12 and raw_bytes[8:12] == b"WEBP"
            return True

    return False


def validate_image_file(file_storage, max_size_bytes: int, allowed_extensions: set) -> Tuple[bool, Optional[str], int, Optional[Dict[str, Any]]]:
    """
    Production-hardened image file validation (3.14.1, 3.14.2, 3.14.3, 3.14.4, 3.14.5):
    - Rejects empty, missing, or null-byte payloads
    - Enforces whitelisted extensions (no dangerous double extensions)
    - Validates binary magic byte headers (MIME authenticity)
    - Enforces hard file-size quota (10 MB limit)
    - Verifies image structure and dimensions with PIL
    """
    if file_storage is None or not getattr(file_storage, "filename", None):
        return False, "No image file provided in request.", 400, None

    filename = sanitize_filename(file_storage.filename)
    if not filename:
        return False, "Invalid, unsafe, or corrupted filename rejected for security.", 400, None

    extension = filename.rsplit(".", 1)[-1].lower()
    if extension not in allowed_extensions:
        allowed_str = ", ".join(sorted(allowed_extensions)).upper()
        return False, f"Unsupported file format '.{extension}'. Allowed formats: {allowed_str}.", 400, None

    # Check file size by seeking (3.14.4)
    file_storage.seek(0, os.SEEK_END)
    file_size = file_storage.tell()
    file_storage.seek(0)

    if file_size > max_size_bytes:
        max_mb = max_size_bytes / (1024 * 1024)
        return False, f"File size ({file_size / (1024 * 1024):.1f} MB) exceeds maximum allowed limit of {max_mb:.0f} MB.", 413, None

    if file_size == 0:
        return False, "The uploaded file is empty (0 bytes).", 400, None

    raw_bytes = file_storage.read()
    file_storage.seek(0)

    # Validate Magic Bytes (3.14.3)
    if not verify_magic_bytes(raw_bytes, extension):
        return False, f"File content does not match genuine {extension.upper()} image signature.", 400, None

    # Verify image integrity with PIL
    try:
        with Image.open(io.BytesIO(raw_bytes)) as img:
            img_format = img.format or extension.upper()
            width, height = img.size
            img.verify()

        info = {
            "filename": filename,
            "extension": extension,
            "format": img_format,
            "width": width,
            "height": height,
            "size_bytes": file_size,
            "size_kb": round(file_size / 1024, 1)
        }
        return True, None, 200, info
    except Exception as e:
        return False, f"Invalid or corrupted image content: {str(e)}", 400, None


def validate_caption_params(form_data: Dict[str, Any]) -> Tuple[str, int, bool]:
    """
    Sanitize and validate caption parameters against injection (3.14.1).
    """
    valid_styles = {"standard", "short", "detailed", "creative"}
    raw_style = form_data.get("style", "standard")
    # Sanitize style parameter to alphanumeric only
    clean_style = re.sub(r"[^a-zA-Z0-9_-]", "", str(raw_style)).lower().strip() if raw_style else "standard"
    if clean_style not in valid_styles:
        clean_style = "standard"

    raw_count = form_data.get("count", form_data.get("num_captions", 1))
    try:
        count = max(1, min(int(raw_count), 3))
    except (ValueError, TypeError):
        count = 1

    raw_regen = form_data.get("regenerate", "false")
    is_regenerate = str(raw_regen).lower() in ("true", "1", "yes")

    return clean_style, count, is_regenerate
