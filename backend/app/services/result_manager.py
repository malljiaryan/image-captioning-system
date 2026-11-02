import os
import json
import uuid
import time
import threading
from typing import Dict, Any, List, Optional
from datetime import datetime

DATA_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "caption_history.json"))
_storage_lock = threading.Lock()


class ResultManager:
    """
    Thread-safe persistent result management service for image captions.
    Stores and manages caption metadata, AI analysis, and evaluation history.
    """

    def __init__(self, storage_path: str = DATA_FILE):
        self.storage_path = storage_path
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        if not os.path.exists(self.storage_path):
            self._write_store([])

    def _read_store(self) -> List[Dict[str, Any]]:
        try:
            if not os.path.exists(self.storage_path):
                return []
            with open(self.storage_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def _write_store(self, items: List[Dict[str, Any]]) -> None:
        temp_file = f"{self.storage_path}.tmp"
        with open(temp_file, "w", encoding="utf-8") as f:
            json.dump(items, f, indent=2, ensure_ascii=False)
        if os.path.exists(self.storage_path):
            os.remove(self.storage_path)
        os.rename(temp_file, self.storage_path)

    def save_result(
        self,
        caption: str,
        captions: List[Dict[str, Any]],
        style: str,
        image_metadata: Dict[str, Any],
        confidence: Dict[str, Any],
        text_stats: Dict[str, Any],
        model_metadata: Dict[str, Any],
        inference_time_ms: float,
        semantics: Optional[Dict[str, Any]] = None,
        quality_evaluation: Optional[Dict[str, Any]] = None,
        is_regenerated: bool = False
    ) -> Dict[str, Any]:
        """Save a new caption generation result and return stored entity."""
        with _storage_lock:
            store = self._read_store()

            result_id = f"cap_{uuid.uuid4().hex[:10]}"
            now = datetime.now()
            record = {
                "id": result_id,
                "caption": caption,
                "captions": captions,
                "style": style,
                "image_metadata": image_metadata,
                "confidence": confidence,
                "text_stats": text_stats,
                "model_metadata": model_metadata,
                "inference_time_ms": inference_time_ms,
                "semantics": semantics or {},
                "quality_evaluation": quality_evaluation or {},
                "is_regenerated": is_regenerated,
                "created_at": now.isoformat(),
                "display_date": now.strftime("Today · %I:%M %p")
            }

            # Prepend newest first
            store.insert(0, record)
            # Retain maximum 500 items to avoid unbounded growth
            if len(store) > 500:
                store = store[:500]

            self._write_store(store)
            return record

    def get_all(self, limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        """Retrieve paginated caption history."""
        with _storage_lock:
            store = self._read_store()
            total = len(store)
            items = store[offset : offset + limit]
            return {
                "total": total,
                "limit": limit,
                "offset": offset,
                "count": len(items),
                "items": items
            }

    def get_by_id(self, result_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific caption record by its ID."""
        with _storage_lock:
            store = self._read_store()
            for item in store:
                if str(item.get("id")) == str(result_id):
                    return item
            return None

    def delete_by_id(self, result_id: str) -> bool:
        """Delete a caption record by ID."""
        with _storage_lock:
            store = self._read_store()
            initial_len = len(store)
            store = [item for item in store if str(item.get("id")) != str(result_id)]
            if len(store) < initial_len:
                self._write_store(store)
                return True
            return False

    def clear_all(self) -> int:
        """Clear all stored caption history and return deleted count."""
        with _storage_lock:
            store = self._read_store()
            count = len(store)
            self._write_store([])
            return count

    def count(self) -> int:
        """Return total count of saved caption results."""
        with _storage_lock:
            return len(self._read_store())


# Singleton instance
result_manager = ResultManager()
