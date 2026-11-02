import json
import uuid
import logging
from typing import Dict, Any, List, Optional
from sqlalchemy.exc import SQLAlchemyError

from app.database import db
from app.models import ImageMetadata, CaptionResult

logger = logging.getLogger(__name__)


class DatabaseService:
    """
    Database Service implementing all CRUD operations for Image Captioning System.
    Handles transactions, relationship cascades, and resilient rollback on errors.
    """

    def save_caption_with_image(
        self,
        image_info: Dict[str, Any],
        caption_text: str,
        captions_list: List[Dict[str, Any]],
        style: str = "standard",
        confidence: Optional[Dict[str, Any]] = None,
        text_stats: Optional[Dict[str, Any]] = None,
        inference_time_ms: float = 0.0,
        semantics: Optional[Dict[str, Any]] = None,
        quality_eval: Optional[Dict[str, Any]] = None,
        is_regenerated: bool = False
    ) -> Dict[str, Any]:
        """
        Store image metadata and generated caption in database (3.13.7, 3.13.8, 3.13.9).
        Maintains relational integrity between ImageMetadata and CaptionResult.
        """
        try:
            filename = image_info.get("filename", "unknown.jpg")
            
            # Check if an ImageMetadata record already exists for this filename (e.g. during regeneration)
            img_record = ImageMetadata.query.filter_by(filename=filename).first()
            if not img_record:
                img_uuid = image_info.get("image_uuid") or str(uuid.uuid4())
                img_record = ImageMetadata(
                    image_uuid=img_uuid,
                    filename=filename,
                    original_name=image_info.get("original_name") or filename,
                    width=image_info.get("width"),
                    height=image_info.get("height"),
                    format=image_info.get("format"),
                    size_kb=image_info.get("size_kb")
                )
                db.session.add(img_record)
                db.session.flush()  # Flush to obtain img_record.id

            conf = confidence or {"score": 85.0, "level": "High"}
            stats = text_stats or {"word_count": len(caption_text.split()), "char_count": len(caption_text)}
            qual = quality_eval or {"composite_score": 85.0, "grade": "High Quality"}

            unique_caption_id = f"cap_{uuid.uuid4().hex[:10]}"
            caption_record = CaptionResult(
                caption_id=unique_caption_id,
                image_id=img_record.id,
                caption_text=caption_text,
                style=style,
                confidence_score=float(conf.get("score", 85.0)),
                confidence_level=str(conf.get("level", "High")),
                word_count=int(stats.get("word_count", 0)),
                char_count=int(stats.get("char_count", 0)),
                inference_time_ms=float(inference_time_ms),
                semantics_json=json.dumps(semantics or {}, ensure_ascii=False),
                quality_score=float(qual.get("composite_score", 85.0)),
                quality_grade=str(qual.get("grade", "High Quality")),
                is_regenerated=is_regenerated
            )

            db.session.add(caption_record)
            db.session.commit()

            result_dict = caption_record.to_dict()
            result_dict["captions"] = captions_list or [{"id": 1, "caption": caption_text, "confidence": conf, "text_stats": stats}]
            return result_dict

        except SQLAlchemyError as err:
            db.session.rollback()  # 3.13.13 Error handling & rollback
            logger.error(f"[DB Error] Failed to save caption with image: {err}", exc_info=True)
            raise err

    def get_caption_by_id(self, caption_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific caption by its unique caption_id (3.13.11)."""
        try:
            record = CaptionResult.query.filter_by(caption_id=str(caption_id)).first()
            return record.to_dict() if record else None
        except SQLAlchemyError as err:
            logger.error(f"[DB Error] get_caption_by_id failed: {err}")
            return None

    def get_all_captions(self, limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        """Retrieve paginated caption history ordered newest first (3.13.10, 3.13.11)."""
        try:
            query = CaptionResult.query.order_by(CaptionResult.created_at.desc())
            total = query.count()
            records = query.offset(offset).limit(limit).all()
            return {
                "total": total,
                "limit": limit,
                "offset": offset,
                "count": len(records),
                "items": [r.to_dict() for r in records]
            }
        except SQLAlchemyError as err:
            logger.error(f"[DB Error] get_all_captions failed: {err}")
            return {"total": 0, "limit": limit, "offset": offset, "count": 0, "items": []}

    def delete_caption_by_id(self, caption_id: str) -> bool:
        """Delete an individual caption record by its ID (3.13.12)."""
        try:
            record = CaptionResult.query.filter_by(caption_id=str(caption_id)).first()
            if not record:
                return False

            image_id = record.image_id
            db.session.delete(record)
            db.session.flush()

            # Clean up parent ImageMetadata if no other captions reference it
            remaining = CaptionResult.query.filter_by(image_id=image_id).count()
            if remaining == 0:
                parent_img = db.session.get(ImageMetadata, image_id)
                if parent_img:
                    db.session.delete(parent_img)

            db.session.commit()
            return True
        except SQLAlchemyError as err:
            db.session.rollback()
            logger.error(f"[DB Error] delete_caption_by_id failed: {err}")
            return False

    def clear_all_captions(self) -> int:
        """Clear all captions and image metadata from database (3.13.12)."""
        try:
            count = CaptionResult.query.count()
            CaptionResult.query.delete()
            ImageMetadata.query.delete()
            db.session.commit()
            return count
        except SQLAlchemyError as err:
            db.session.rollback()
            logger.error(f"[DB Error] clear_all_captions failed: {err}")
            return 0

    def get_count(self) -> int:
        """Return total caption count in database."""
        try:
            return CaptionResult.query.count()
        except SQLAlchemyError:
            return 0


# Singleton database service instance
db_service = DatabaseService()
