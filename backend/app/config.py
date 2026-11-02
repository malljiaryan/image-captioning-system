import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-later")

    UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

    # ---------------------------------------------------------------------------
    # Database Configuration (3.13.5 SQLAlchemy, 3.13.6 MySQL / SQLite)
    # ---------------------------------------------------------------------------
    # MySQL Environment Parameters
    MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
    MYSQL_PORT = os.getenv("MYSQL_PORT", "3306")
    MYSQL_USER = os.getenv("MYSQL_USER", "root")
    MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
    MYSQL_DB = os.getenv("MYSQL_DB", "image_captioning_db")

    # Custom DATABASE_URL or MySQL connection string
    custom_db_uri = os.getenv("DATABASE_URL") or os.getenv("MYSQL_URL")
    if custom_db_uri:
        SQLALCHEMY_DATABASE_URI = custom_db_uri
    elif os.getenv("USE_MYSQL", "false").lower() in ("true", "1"):
        # Format for PyMySQL driver
        SQLALCHEMY_DATABASE_URI = (
            f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}?charset=utf8mb4"
        )
    else:
        # Default local persistent SQLite database
        default_sqlite_path = os.path.join(DATA_DIR, "image_captioning.db")
        SQLALCHEMY_DATABASE_URI = f"sqlite:///{default_sqlite_path}"

    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        "pool_pre_ping": True,
        "pool_recycle": 280
    }