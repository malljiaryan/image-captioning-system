import json
from datetime import datetime
from app.database import db


class ImageMetadata(db.Model):
    """
    Image Metadata Model (3.13.2, 3.13.8)
    Stores physical image attributes and file tracking references.
    """
    __tablename__ = "image_metadata"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    image_uuid = db.Column(db.String(36), unique=True, nullable=False, index=True)
    filename = db.Column(db.String(255), nullable=False)
    original_name = db.Column(db.String(255), nullable=True)
    width = db.Column(db.Integer, nullable=True)
    height = db.Column(db.Integer, nullable=True)
    format = db.Column(db.String(20), nullable=True)
    size_kb = db.Column(db.Float, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    # 1-to-many relationship: One image can have multiple caption results (variations / regenerated)
    # Cascade delete ensures deleting an image deletes all its associated captions
    captions = db.relationship(
        "CaptionResult",
        backref="image",
        cascade="all, delete-orphan",
        lazy=True,
        passive_deletes=True
    )

    def to_dict(self):
        return {
            "id": self.id,
            "image_uuid": self.image_uuid,
            "filename": self.filename,
            "original_name": self.original_name,
            "width": self.width,
            "height": self.height,
            "format": self.format,
            "size_kb": self.size_kb,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class CaptionResult(db.Model):
    """
    Caption Result Model (3.13.3, 3.13.9, 3.13.10)
    Stores natural language captions, confidence metrics, semantics, and quality scores.
    """
    __tablename__ = "caption_result"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    caption_id = db.Column(db.String(50), unique=True, nullable=False, index=True)

    # Foreign Key relationship (3.13.4)
    image_id = db.Column(
        db.Integer,
        db.ForeignKey("image_metadata.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    caption_text = db.Column(db.Text, nullable=False)
    style = db.Column(db.String(30), default="standard", nullable=False)
    confidence_score = db.Column(db.Float, nullable=True)
    confidence_level = db.Column(db.String(20), nullable=True)
    word_count = db.Column(db.Integer, nullable=True)
    char_count = db.Column(db.Integer, nullable=True)
    inference_time_ms = db.Column(db.Float, nullable=True)

    # Stored as serialized JSON string for rich scene intelligence
    semantics_json = db.Column(db.Text, nullable=True)
    quality_score = db.Column(db.Float, nullable=True)
    quality_grade = db.Column(db.String(30), nullable=True)
    is_regenerated = db.Column(db.Boolean, default=False, nullable=False)

    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)  # 3.13.10

    def get_semantics(self):
        try:
            return json.loads(self.semantics_json) if self.semantics_json else {}
        except Exception:
            return {}

    def to_dict(self):
        created_dt = self.created_at or datetime.utcnow()
        return {
            "id": self.caption_id,
            "caption": self.caption_text,
            "style": self.style,
            "confidence": {
                "score": self.confidence_score or 0.0,
                "level": self.confidence_level or "Normal"
            },
            "text_stats": {
                "word_count": self.word_count or len(self.caption_text.split()),
                "char_count": self.char_count or len(self.caption_text)
            },
            "inference_time_ms": self.inference_time_ms or 0.0,
            "semantics": self.get_semantics(),
            "quality_evaluation": {
                "composite_score": self.quality_score or 0.0,
                "grade": self.quality_grade or "Standard"
            },
            "image_metadata": self.image.to_dict() if self.image else {},
            "is_regenerated": self.is_regenerated,
            "created_at": created_dt.isoformat(),
            "display_date": created_dt.strftime("Today · %I:%M %p")
        }
