# VisionCaption AI - Intelligent Deep Learning Image Captioning System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Framework](https://img.shields.io/badge/Backend-Flask%203.1-green.svg)
![AI Model](https://img.shields.io/badge/AI%20Model-Salesforce%20BLIP-orange.svg)
![Database](https://img.shields.io/badge/Database-SQLAlchemy%20%7C%20MySQL%20%7C%20SQLite-blueviolet.svg)
![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)

An enterprise-grade, end-to-end AI image captioning platform built with **Salesforce BLIP** (Bootstrapping Language-Image Pre-training), **PyTorch**, **Flask**, **SQLAlchemy**, and modern vanilla **HTML5 / CSS3 / JavaScript**.

---

## 🌟 Key Features

1. **Deep Learning Vision-Language Inference**:
   - Uses `Salesforce/blip-image-captioning-base` (247 million parameters) for natural language image captioning.
   - Singleton model loader with thread-safe concurrency lock and startup warmup graph pre-compilation.
2. **Multi-Style Caption Generation**:
   - **✦ Standard**: Balanced, natural description of visual elements.
   - **⚡ Short**: Concise, punchy summaries (low token budget).
   - **🔍 Detailed**: Rich, descriptive captions covering foreground and background.
   - **🎨 Creative**: Stochastic nucleus sampling with temperature variation for poetic/expressive descriptions.
3. **Multiple Candidate Variations**:
   - Generate up to 3 candidate captions per image with interactive selection cards and confidence scores.
4. **Semantic Scene Intelligence**:
   - Automated visual probing extracting **Major Objects**, **Scene & Setting**, **Actions/Activities**, and **Visual Attributes**.
5. **Caption Quality & NLP Benchmarking**:
   - Reference-free Composite Quality Score (0–100) and Quality Grade.
   - Reference-based evaluation against ground truth using **BLEU-1..4**, **ROUGE-1/2/L**, **METEOR**, and **Semantic Cosine Similarity**.
6. **Persistent Relational Storage**:
   - Dual-engine database architecture: **MySQL** (production) and **SQLite** (local zero-config fallback).
   - Normalized relational schema with cascading deletions and foreign key integrity.
7. **Production Security & Hardening**:
   - Strict filename sanitization and directory traversal prevention.
   - Binary magic byte MIME verification (`\xFF\xD8\xFF` for JPEG, `\x89PNG` for PNG, `RIFF...WEBP`).
   - 10 MB maximum upload quota enforcement (HTTP 413).
   - Universal CORS and HTTP Security Headers (`nosniff`, `SAMEORIGIN`, `1; mode=block`).
8. **Export & Sharing**:
   - Instant export of comprehensive caption reports to **TXT** and **JSON** formats.
   - One-click clipboard copy.

---

## 🏗️ System Architecture

```
[Browser Client]
  │
  ├─► caption.html  (Upload, Style Select, Multi-Variations, Semantics, NLP Benchmarks)
  ├─► history.html  (Dynamic History Loading, Delete, Clear All)
  │
  ▼
[Flask API Gateway (Port 5000)]
  │
  ├─► Middleware: API Logging, Request Timer, Security Headers (nosniff, SAMEORIGIN)
  ├─► Validators: Magic Bytes, 10MB Quota, Sanitized Filenames, Injection Protection
  │
  ├─► AI Inference Engine:
  │     ├─► Salesforce BLIP Base (PyTorch CPU / CUDA)
  │     ├─► Preprocessing: Bicubic Downscaling (MAX_IMAGE_DIM = 1200)
  │     ├─► Semantic Scene Analyzer (Visual Probing)
  │     └─► NLP Quality Evaluator (BLEU, ROUGE, METEOR, Similarity)
  │
  └─► Database Layer (SQLAlchemy ORM):
        ├─► ImageMetadata Table (Width, Height, Format, Size, File tracking)
        └─► CaptionResult Table (Captions, Style, Confidence, Semantics, Quality, Timestamps)
```

---

## 🚀 Installation & Setup Guide (From ZIP Archive or Git)

Follow the dedicated step-by-step instructions below for your operating system:

---

### 🍏 macOS Setup Guide (Step-by-Step)

Works smoothly on all Mac computers (**Apple Silicon M1 / M2 / M3 / M4** and **Intel Macs**).

#### Step 1: Verify / Install Python 3.10+
Open **Terminal** (press `Cmd + Space`, type `Terminal`, and press `Enter`).  
Check your Python version:
```bash
python3 --version
```
If Python is not installed or older than 3.10:
- **Option A (Recommended)**: Download and install the official macOS installer (.pkg) from [python.org/downloads](https://www.python.org/downloads/).
- **Option B (Homebrew)**: If you use Homebrew, run:
  ```bash
  brew install python@3.10
  ```

#### Step 2: Extract the ZIP & Navigate into Project
1. Double-click **`VisionCaption-AI.zip`** in Finder to extract it.
2. In your Terminal, navigate to the extracted directory:
```bash
# Example if extracted in Downloads:
cd ~/Downloads/"Image captioning system"

# OR if extracted on Desktop:
cd ~/Desktop/"Image captioning system"
```

#### Step 3: Create and Activate Virtual Environment
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```
*(Once activated, your terminal prompt will display `(venv)`).*

#### Step 4: Install Required Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
*(This installs Flask, PyTorch, Transformers, Pillow, SQLAlchemy, and associated utilities).*

#### Step 5: Launch the Full-Stack Application
```bash
python3 run.py
```

You will see the startup banner:
```text
============================================================
  VisionCaption AI - Deep Learning Image Captioning Platform
============================================================
  * Web Application:   http://127.0.0.1:5000
  * Caption Studio:    http://127.0.0.1:5000/caption
  * Caption History:   http://127.0.0.1:5000/history
  * Architecture/Docs: http://127.0.0.1:5000/about
  * REST API Root:     http://127.0.0.1:5000/api
============================================================
```

#### Step 6: Open in Safari or Chrome
Open your web browser and go to:
👉 **[http://127.0.0.1:5000](http://127.0.0.1:5000)** (or [http://localhost:5000](http://localhost:5000))

- **Caption Studio**: `http://127.0.0.1:5000/caption` (Upload images, choose styles, generate AI captions)
- **Caption History**: `http://127.0.0.1:5000/history` (View, search, and manage caption history)
- **Architecture & Docs**: `http://127.0.0.1:5000/about`

> **🍎 Important Note for macOS Users (AirPlay Port 5000 Conflict)**:  
> On macOS Monterey, Ventura, Sonoma, and Sequoia, macOS uses port 5000 by default for Apple's **AirPlay Receiver**.  
> If `python3 run.py` shows `Address already in use: 5000`, turn off AirPlay Receiver:  
> 1. Open **System Settings** (or System Preferences).  
> 2. Go to **General** ➔ **AirDrop & AirPlay**.  
> 3. Turn **AirPlay Receiver** to **OFF**.  
> Alternatively, free the port by running: `lsof -ti:5000 | xargs kill -9`.

---

### 🪟 Windows Setup Guide (Step-by-Step)

#### Step 1: Install Python 3.10+
- Download and install Python from [python.org/downloads](https://www.python.org/downloads/).
- **CRITICAL**: During setup, check the box **"Add python.exe to PATH"**.

#### Step 2: Extract the ZIP & Open Terminal
1. Right-click **`VisionCaption-AI.zip`** ➔ **Extract All...**
2. Open **PowerShell** or **Command Prompt** inside the extracted folder:
```powershell
cd "Image captioning system"
```

#### Step 3: Create and Activate Virtual Environment
```powershell
# Create virtual environment
python -m venv venv

# Activate in PowerShell:
.\venv\Scripts\Activate.ps1

# OR activate in Command Prompt (cmd.exe):
venv\Scripts\activate.bat
```
*(If PowerShell displays a script execution policy error, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and then run `.\venv\Scripts\Activate.ps1` again).*

#### Step 4: Install Required Dependencies
```powershell
pip install -r requirements.txt
```

#### Step 5: Launch the Application
```powershell
python run.py
```

#### Step 6: Open in Browser
Visit **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in Chrome, Edge, or Firefox.

---

### 🐧 Linux Setup Guide (Ubuntu / Debian / Fedora)

```bash
# 1. Install Python and venv package
sudo apt update && sudo apt install -y python3 python3-pip python3-venv

# 2. Extract and navigate to folder
unzip VisionCaption-AI.zip
cd "Image captioning system"

# 3. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Launch application
python3 run.py
```
Open **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your browser.

---

### ⚡ Unified Full-Stack Architecture Note (All Platforms)
`python run.py` automatically serves **both** the Flask REST API and all frontend web pages on port 5000. No separate Node.js, Nginx, or secondary server is needed on any platform!

---

### 🧠 First-Run Model Download & Warmup
- **Automatic Download**: On the very first run, the system automatically downloads the pre-trained Salesforce BLIP model (`Salesforce/blip-image-captioning-base`, ~990 MB) directly from Hugging Face Hub. This happens only once and is safely cached locally (`~/.cache/huggingface/hub/`).
- **Instant Warmup**: During server startup, the model is pre-warmed in a background thread so your first caption generates smoothly.

---

### 🗄️ Database Configuration
- **Zero-Config Default (SQLite)**: Out of the box, the application creates and uses a local SQLite database at `backend/data/image_captioning.db`. No installation or credentials needed.
- **Optional MySQL Mode**: To use MySQL, copy `.env.example` to `.env` and configure your credentials:
  ```env
  USE_MYSQL=true
  MYSQL_HOST=localhost
  MYSQL_PORT=3306
  MYSQL_USER=root
  MYSQL_PASSWORD=your_password
  MYSQL_DB=image_captioning_db
  ```

---

## 📡 REST API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the web application homepage |
| `GET` | `/caption` | Serves the Caption Studio web interface |
| `GET` | `/history` | Serves the Caption History web interface |
| `GET` | `/about` | Serves Architecture & Documentation page |
| `GET` | `/api` | REST API service discovery and endpoints index |
| `GET` | `/api/health` | Service health status and memory usage |
| `GET` | `/api/performance/stats` | Live runtime performance metrics and device info |
| `POST` | `/api/caption` | Upload image and generate natural language caption(s) |
| `POST` | `/api/caption/regenerate` | Regenerate caption using stored result ID or image |
| `GET` | `/api/caption/<id>` | Retrieve specific caption result by ID |
| `GET` | `/api/history` | Retrieve paginated caption history (`limit`, `offset`) |
| `DELETE` | `/api/history/<id>` | Delete individual caption record |
| `DELETE` | `/api/history` | Clear all caption history |
| `POST` | `/api/evaluate` | Benchmark candidate caption against reference caption |

---

## 🧪 Verification & Automated Testing

To verify that all components are functioning properly on your system, run the test suites inside your activated virtual environment:

```bash
# 1. Complete System Acceptance Test Suite (Recommended)
python -m unittest backend/tests/test_complete_system.py

# 2. API Integration Test Suite
python -m unittest backend/tests/test_api_integration.py

# 3. Database & Relational Persistence Test Suite
python -m unittest backend/tests/test_database.py

# 4. Security & Production Hardening Test Suite
python -m unittest backend/tests/test_security.py

# 5. Performance & Optimization Test Suite
python -m unittest backend/tests/test_performance.py
```

---

## 🔧 Troubleshooting & FAQ

### Q1: "Port 5000 is already in use"
- **On macOS (AirPlay Receiver Conflict)**:
  Apple uses port 5000 for AirPlay by default. Go to **System Settings ➔ General ➔ AirDrop & AirPlay ➔ AirPlay Receiver ➔ Turn OFF**.  
  Or kill the process from Terminal:
  ```bash
  lsof -ti:5000 | xargs kill -9
  ```
- **On Windows**:
  ```powershell
  Get-NetTCPConnection -LocalPort 5000 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
  ```
- **On Linux**:
  ```bash
  fuser -k 5000/tcp
  ```

### Q2: "Command not found: python" (macOS / Linux)
On macOS and Linux, Python 3 is invoked with `python3` and pip with `pip3`:
```bash
python3 -m venv venv
source venv/bin/activate
python run.py
```

### Q3: "Running scripts is disabled on this system" (Windows PowerShell)
If activating the virtual environment fails with an ExecutionPolicy error:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```
Or use the standard Windows Command Prompt (`cmd.exe`):
```cmd
venv\Scripts\activate.bat
```

### Q4: Does this work on Apple Silicon (M1/M2/M3/M4) Macs?
Yes! PyTorch and Transformers run natively on Apple Silicon ARM64 chips. The model runs fast on CPU and utilizes macOS optimized BLAS / Accelerate libraries out of the box.

### Q5: Does this work on CPU or does it require a GPU?
- The system is optimized to run on standard **CPUs** with multi-threading optimizations, greedy decoding, and image downscaling guards.
- If an NVIDIA GPU with CUDA is detected, PyTorch will automatically utilize the GPU for accelerated inference.

---

## 📦 How to Create the ZIP File for Sharing

If you are packaging this project into a ZIP file to share with someone else, follow these guidelines:

### ✅ What to INCLUDE in the ZIP:
- `backend/` (all source code, models, routes, services, database schema)
- `frontend/` (HTML, CSS, JavaScript files)
- `run.py` (entry point script)
- `requirements.txt` (dependencies list)
- `README.md` (setup guide)
- `.env.example` (sample configuration)
- `uploads/` (empty folder or sample test image)

### ❌ What to EXCLUDE from the ZIP:
- **`venv/` (DO NOT INCLUDE)**: The virtual environment is 4+ GB in size and contains machine-specific paths and binaries. The recipient will generate their own clean virtual environment in 1 command (`python -m venv venv`).
- **`__pycache__/` and `*.pyc`**: Compiled Python bytecode files.
- **`.git/`**: Git revision history (if sharing just source code).
- **`.pytest_cache/`** (temporary test cache).

---

## 📄 License
MIT License. Developed with Google DeepMind Advanced Agentic AI.