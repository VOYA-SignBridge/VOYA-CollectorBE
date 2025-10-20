Recommended developer setup (prefer Python 3.11)

Why: Some binary packages used in this project (mediapipe, opencv, etc.) are more reliably available for Python 3.11/3.10. Using Python 3.11 reduces installation issues on Windows.

Create a dedicated venv with Python 3.11 (PowerShell):

```powershell
# replace <path-to-python311> with your python 3.11 executable if not on PATH
# Example if python3.11 is available on PATH: py -3.11 -m venv env311
py -3.11 -m venv .venv311
.\.venv311\Scripts\Activate.ps1

# from repository root (where backend/ sits)
cd d:\VOYA_Code\VOYA_Collect_dataset\sign_dataset_backend
pip install --upgrade pip
pip install -r backend\requirements.txt
```

Quick import test (after venv activation and installing deps):

```powershell
# run from repository root
python -c "from backend.app.routers.dataset_exporter import export_dataset; print('dataset_exporter imported OK')"
```

If you see import errors mentioning mediapipe or opencv wheels, try one of:
- Install a different supported wheel version (check PyPI for available builds)
- Use Python 3.11 (recommended) if you're on 3.13
- If a package must be built from source, follow package-specific instructions (often difficult on Windows)

Notes:
- I added `tqdm` to `backend/requirements.txt` because `backend/app/processing/utils.py` imports it.
- Ensure you run Uvicorn from the `backend` directory or set PYTHONPATH so `app` package imports resolve correctly, e.g.:

```powershell
cd backend
python -m uvicorn app.main:app --reload
```

This document is intentionally short — tell me if you want me to add automated checks or a small script that verifies sample shapes before creating memmap files.