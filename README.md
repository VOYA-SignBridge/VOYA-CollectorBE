# VOYA - Sign Dataset Backend (sign_dataset_backend)

This repository contains the backend service and processing pipeline used to collect, validate, augment, and export sign language training data with support for dialect variations.

This README covers development setup, Docker/docker-compose usage, running the worker, dataset export, dialect support, and deployment notes.

## Table of contents
- Prerequisites
- Local development (virtualenv)
- Docker / docker-compose (recommended for production-like local runs)
- Running the worker (Celery)
- Dialect support
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
pip install -r backend\requirements.txt
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
docker compose logs -f worker
```

3. Test API endpoints

```powershell
# List labels
curl http://localhost:8000/dataset/labels

# Check OpenAPI docs
# Open browser: http://localhost:8000/docs
```

4. Stop services

```powershell
docker compose down
```

Notes:
- The compose file mounts local `dataset/` folder into the backend container; ensure `dataset/` exists (it is .gitignored).
- If you change Python dependencies, rebuild images with `docker compose build backend worker` or `--build` above.
- Build context optimized with `.dockerignore` to avoid transferring large dataset files.

## Running the worker (Celery)

If you run via docker-compose the worker will be started automatically. To run locally (outside Docker):

```powershell
.\env\Scripts\Activate.ps1
# start redis locally first
celery -A backend.app.worker worker --loglevel=info
```

Celery configuration is in `backend/app/worker.py`.

## Dialect support

The backend now supports dialect variations for sign language data collection. Both video and camera uploads can include dialect metadata.

### Video upload with dialect (FormData)

```bash
curl -X POST http://localhost:8000/upload/video \
  -F "file=@sample_video.mp4" \
  -F "user=testuser" \
  -F "label=xin chào" \
  -F "dialect=Bắc"
```

### Camera upload with dialect (JSON)

```bash
curl -X POST http://localhost:8000/upload/camera \
  -H "Content-Type: application/json" \
  -d '{
    "user": "testuser",
    "label": "xin chào", 
    "dialect": "Nam",
    "session_id": "test123",
    "frames": [
      {"timestamp": 0, "landmarks": [/* 1605 keypoint values */]},
      {"timestamp": 33, "landmarks": [/* 1605 keypoint values */]}
    ]
  }'
```

### Dialect metadata storage

- Dialect is stored in `dataset/samples.csv` as a new column
- Also saved in individual JSON metadata files alongside NPZ files
- Available for filtering and analytics in dataset export
- Nullable field - backward compatible with existing data

### Supported dialect values

Common values: `"Bắc"`, `"Trung"`, `"Nam"`, or custom dialect names
- Max length: 128 characters
- Optional field - can be empty string or omitted

## Exporting dataset (memmap)

Use the dataset exporter API to validate and export the processed features into memmap files suitable for training.

Example (local):

```powershell
curl -X POST "http://localhost:8000/dataset/export?fix=true"
```

- `fix=true` will attempt to auto-fix common dataset issues (pad/truncate to T=60, shape checks).
- Exported files are saved under `dataset/processed/memmap/` as: `dataset_X.dat`, `dataset_y.dat`, and `dataset_meta.json`.
- Dataset metadata now includes dialect distribution statistics.

If you prefer a programmatic call, check `backend/app/routers/dataset_exporter.py`.

## CORS Support

Backend includes CORS middleware to allow frontend applications to make requests:

- **Allowed origins**: `http://localhost:5173`, `http://localhost:3000`, `http://127.0.0.1:5173`
- **Allowed methods**: All HTTP methods
- **Credentials**: Supported
- **Headers**: All headers allowed

For production, update allowed origins in `backend/app/main.py` to match your frontend domain.

## Examples

Below are quick examples for common operations (assumes backend running at http://localhost:8000):

- Health check / API test

```powershell
# Test API is responding
Invoke-RestMethod -Uri 'http://localhost:8000/dataset/labels' -UseBasicParsing | ConvertTo-Json

# Or view OpenAPI docs in browser
# http://localhost:8000/docs
```

- Camera upload (JSON keypoints payload with dialect)

```powershell
curl -X POST http://localhost:8000/upload/camera \
	-H "Content-Type: application/json" \
	-d @sample_upload.json
```

Where `sample_upload.json` includes dialect field. See `camera_upload_test.html` for an example payload.

- Export dataset (with auto-fix)

```powershell
curl -X POST "http://localhost:8000/dataset/export?fix=true"
```

- Check exported files (on server)

```powershell
ls dataset\processed\memmap
cat dataset\processed\memmap\dataset_meta.json
```

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
- Added `.dockerignore` to reduce build context (excludes dataset/, env/, etc.)

### Build time optimization

- Initial Docker builds may take 4-6 minutes due to heavy ML packages (OpenCV, MediaPipe, JAX).
- Use Docker layer caching for faster rebuilds.
- For development, consider splitting requirements into dev/prod versions.

## Database migration

If migrating from CSV to proper database:

```sql
-- Add dialect column to samples table
ALTER TABLE samples ADD COLUMN dialect VARCHAR(128) NULL;
CREATE INDEX idx_samples_dialect ON samples(dialect);
```

Currently using CSV files in `dataset/samples.csv` with dialect column added.

## Useful scripts & tests

- `tools/train_baseline.py` — small PyTorch baseline trainer (smoke test / baseline)
- `tools/torch_dataset.py` — PyTorch Dataset with on-the-fly augmentation
- `tools/test_normalize.py` — test normalize_sequence behaviour
- `test_camera_upload.py` — integration test for camera uploads (requires backend running)
- `scripts/add_dialect_column.sql` — database migration for dialect support
- `scripts/` — helper scripts to repair dataset metadata and reorganize samples

## Export checklist before training

1. Ensure `dataset/features/*/*.npz` and metadata JSON files are present and validated.
2. Run the exporter API with `fix=true` to normalize shapes and fix common issues.
3. Inspect `dataset/processed/memmap/dataset_meta.json` for class/user/dialect distribution.
4. Verify dialect metadata is properly populated in samples.

## Contributing

- Open PRs against `main` branch. For CI, add tests to the `tests/` folder and update `requirements.txt`.
- When adding new API endpoints, update OpenAPI documentation and include dialect support where relevant.

