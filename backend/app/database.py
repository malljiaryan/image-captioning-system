from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import event
from sqlalchemy.engine import Engine

db = SQLAlchemy()

@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """Ensure SQLite uses WAL mode and busy timeout to avoid concurrency lockups."""
    try:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=10000")
        cursor.close()
    except Exception:
        pass

def init_db(app):
    """Initialize SQLAlchemy with Flask app context and create all tables."""
    db.init_app(app)
    with app.app_context():
        db.create_all()

