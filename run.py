import sys
import os
import threading

# Add backend directory to Python system path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from app import create_app
from app.ai.model import warmup_model

app = create_app()

def _warmup_worker():
    """Pre-warm model in background so the first request is instantaneous."""
    try:
        warmup_model()
    except Exception as e:
        print(f"[BLIP Pre-warmup Notice] {e}")

if __name__ == '__main__':
    # Start background model warmup immediately
    warmup_thread = threading.Thread(target=_warmup_worker, daemon=True)
    warmup_thread.start()

    print("=" * 60)
    print("  VisionCaption AI - Deep Learning Image Captioning Platform")
    print("=" * 60)
    print("  * Web Application:   http://127.0.0.1:5000")
    print("  * Caption Studio:    http://127.0.0.1:5000/caption")
    print("  * Caption History:   http://127.0.0.1:5000/history")
    print("  * Architecture/Docs: http://127.0.0.1:5000/about")
    print("  * REST API Root:     http://127.0.0.1:5000/api")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)