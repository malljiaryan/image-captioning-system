from flask import jsonify
from typing import Any, Optional, Dict


def api_response(
    success: bool = True,
    message: str = "",
    data: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
    status_code: int = 200,
    **extra_fields
):
    """
    Centralized JSON response utility for consistent API structure across all endpoints.
    Format:
    {
        "success": true/false,
        "message": "...",
        "status_code": 200,
        "data": { ... },
        "error": null or "...",
        ...extra_fields
    }
    """
    payload = {
        "success": success,
        "message": message,
        "status_code": status_code
    }

    if data is not None:
        if isinstance(data, dict):
            # Also merge data directly into payload for backward compatibility with existing frontend
            for k, v in data.items():
                if k not in payload:
                    payload[k] = v
            payload["data"] = data
        else:
            payload["data"] = data

    if error is not None:
        payload["error"] = error

    for k, v in extra_fields.items():
        payload[k] = v

    return jsonify(payload), status_code
