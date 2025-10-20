# VOYA - Sign Dataset Backend (sign_dataset_backend)

This repository contains the backend service and processing pipeline used to collect, validate, augment, and export sign language training data.

This README covers development setup, Docker/docker-compose usage, running the worker, dataset export, and deployment notes.

## Table of contents
- Prerequisites
- Local development (virtualenv)
- Docker / docker-compose (recommended for production-like local runs)
- Running the worker (Celery)
- Exporting the dataset (memmap)
- Deployment notes
- Useful scripts

## Prerequisites
- Python 3.10+
- pip
- (Optional) Docker & docker-compose v2+
- (Optional) Redis + Postgres for worker/DB (docker-compose can provide these)

## Local development (virtualenv)
1. Create and activate a virtualenv

```powershell
cd D:\VOYA_Code\VOYA_Collect_dataset\sign_dataset_backend
python -m venv env
.\env\Scripts\Activate.ps1
```

2. Install dependencies

```powershell
pip install -r requirements.txt
```

3. Run the backend (development)

```powershell
# from project root
.\env\Scripts\Activate.ps1
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

The API root is `http://localhost:8000`. Check `backend/app/routers` for available endpoints (upload, dataset exporter, jobs).

## Docker / docker-compose (recommended for local integration)

This repo includes a `docker-compose.yml` at the workspace root. It can stand up the backend, a Celery worker, Redis and Postgres.

1. Build and run services (from workspace root):

```powershell
cd D:\VOYA_Code\VOYA_Collect_dataset\sign_dataset_backend
docker compose up --build -d
```

2. Check logs

```powershell
docker compose logs -f backend
```

3. Stop services

```powershell
docker compose down
```

Notes:
- The compose file mounts local `dataset/` folder into the backend container; ensure `dataset/` exists (it is .gitignored).
- If you change Python dependencies, rebuild images with `docker compose build backend` or `--build` above.

## Running the worker (Celery)

If you run via docker-compose the worker will be started automatically. To run locally (outside Docker):

```powershell
.\env\Scripts\Activate.ps1
# start redis locally first
celery -A backend.app.worker worker --loglevel=info
```

Celery configuration is in `backend/app/worker.py`.

## Exporting dataset (memmap)

Use the dataset exporter API to validate and export the processed features into memmap files suitable for training.

Example (local):

```powershell
curl -X POST "http://localhost:8000/api/dataset/export?fix=true"
```

- `fix=true` will attempt to auto-fix common dataset issues (pad/truncate to T=60, shape checks).
- Exported files are saved under `dataset/processed/memmap/` as: `dataset_X.dat`, `dataset_y.dat`, and `dataset_meta.json`.

If you prefer a programmatic call, check `backend/app/routers/dataset_exporter.py`.

## Deployment notes

- Production deployment should run the backend app behind a reverse proxy (nginx) and use a process manager (systemd/docker-compose/kubernetes).
- Use environment variables for configuration. Example env vars used in code:
	- `DATABASE_URL` (Postgres DSN)
	- `REDIS_URL` (for Celery broker)
	- `APP_ENV` (production/development)

- Secrets: do not store secrets in git. Use secret manager or environment variables.

### Docker image tips

- Use multi-stage builds to produce small images.
- Pin Python and dependency versions in `requirements.txt`.

## Useful scripts & tests

- `tools/train_baseline.py` — small PyTorch baseline trainer (smoke test / baseline)
- `tools/torch_dataset.py` — PyTorch Dataset with on-the-fly augmentation
- `tools/test_normalize.py` — test normalize_sequence behaviour
- `test_camera_upload.py` — integration test for camera uploads (requires backend running)
- `scripts/` — helper scripts to repair dataset metadata and reorganize samples

## Export checklist before training

1. Ensure `dataset/features/*/*.npz` and metadata JSON files are present and validated.
2. Run the exporter API with `fix=true` to normalize shapes and fix common issues.
3. Inspect `dataset/processed/memmap/dataset_meta.json` for class/user distribution.

## Contributing

- Open PRs against `main` branch. For CI, add tests to the `tests/` folder and update `requirements.txt`.

