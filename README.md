# VOYA - Sign Dataset Backend (sign_dataset_backend)

Small README with quick dev and run instructions for the backend service used to collect and process sign language data.

## Prerequisites
- Python 3.10+ (virtualenv recommended)
- pip
- (Optional) Docker & docker-compose for running backend + worker + Redis/Postgres

## Setup (local virtualenv)
```powershell
cd D:\VOYA_Code\VOYA_Collect_dataset\sign_dataset_backend
python -m venv env
.\env\Scripts\Activate.ps1
pip install -r requirements.txt
```

Note: if `requirements.txt` is not present in this folder, check the project root or run `pip install fastapi uvicorn numpy opencv-python` and other dependencies listed in `backend/app`.

## Run backend (development)
```powershell
# Activate venv first
.\env\Scripts\Activate.ps1
# Run using uvicorn
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Run worker (if using Celery)
```powershell
# assuming Redis is running locally
celery -A backend.app.worker worker --loglevel=info
```

## Tests and utilities
- `tools/train_baseline.py` - quick trainer skeleton (PyTorch)
- `tools/test_normalize.py` - test normalize_sequence outputs
- `test_camera_upload.py` - integration test for camera upload (requires backend running)

## Notes
- The repo intentionally excludes raw dataset files from VCS. Place data under `dataset/` (ignored) when needed.
- Check `.gitignore` before adding large files.

If you want, I can:
- add a minimal `requirements.txt` inferred from imports,
- add GitHub actions for CI (lint/test), or
- expand README with docker-compose instructions.
