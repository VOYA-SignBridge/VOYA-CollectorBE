# Hướng dẫn vận hành (Operator/Admin)

Tài liệu này dành cho người **cài đặt, chạy dịch vụ, trỏ model, hỗ trợ user**.

## 1) Tổng quan thành phần

- **Backend**: FastAPI (port 8000)
- **Frontend**: React/Vite (thường port 5173)
- **Model inference**: PyTorch checkpoint `model_best.pth.tar`

Mặc định pipeline hiện tại là **hands-only 126-dim** (FEATURE_DIM=126).

## 2) Chạy backend

### Cách A — Docker Compose (khuyến nghị)

Từ thư mục `sign_dataset_backend`:

```powershell
docker compose up --build -d
```

Kiểm tra:

```powershell
curl http://localhost:8000/health
curl http://localhost:8000/api/inference/model
```

Xem log:

```powershell
docker compose logs -f backend
```

### Cách B — Local venv

```powershell
cd sign_dataset_backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```

## 3) Chạy frontend

Từ thư mục `VOYA-CollectorFE`:

```powershell
npm install
npm run dev
```

Đặt biến môi trường `VITE_API_URL` trỏ về backend (thường `http://localhost:8000`).

## 4) Train PyTorch model (checkpoint + label_map)

### 4.1 Chuẩn bị môi trường train

Từ `sign_dataset_backend`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
```

### 4.2 Chạy train

Ví dụ train theo dialect `vn/common` và xuất ra `models/test_run/`:

```powershell
python tools\train_baseline.py \
  --data-root dataset/features \
  --labels-csv dataset/labels.csv \
  --language vn \
  --dialect common \
  --out-dir models/test_run \
  --epochs 30 \
  --pooling attn
```

Kết quả mong đợi:
- `models/test_run/model_best.pth.tar`
- `models/test_run/label_map.json`

Checkpoint hiện lưu kèm:
- `input_dim` (mặc định 126)
- `pooling` (mean/last/attn)
- `norm_mean`, `norm_std` (normalization per-dim, train-only)

## 5) Trỏ backend sang model mới

### 5.1 Khi chạy local

Set env:
- `INFERENCE_BACKEND=pytorch`
- `MODEL_ARTIFACT_PATH=models/test_run/model_best.pth.tar`
- (optional) `MODEL_LABEL_MAP_PATH=models/test_run/label_map.json`

Restart backend, rồi kiểm tra:

```powershell
curl http://localhost:8000/api/inference/model
```

Bạn nên thấy:
- `backend: pytorch`
- `loaded: true`
- `has_label_map: true`
- `has_norm: true`
- `pooling: attn` (nếu train với `--pooling attn`)

### 5.2 Khi chạy Docker

- Copy/volume mount folder model vào container `/models/...` (theo compose hiện tại).
- Set `MODEL_ARTIFACT_PATH=/models/test_run/model_best.pth.tar`

## 6) Các lỗi hay gặp

### Không tìm thấy sample khi train
- Kiểm tra đường dẫn đúng: `dataset/features/vn/common/<class_uid>_<slug>/*.npz`
- Kiểm tra `dataset/labels.csv` có cột `class_uid` và `class_idx`.

### Val bị thiếu lớp
- Script đã dùng stratified split và best-effort “bù lớp”, nhưng nếu lớp quá ít sample hoặc metadata nhóm trống, cần thu thêm data.

### Model "đứng yên cũng đoán"
- Đã có các cơ chế gating/temperature ở backend và gating ở frontend.
- Nếu vẫn nặng, ưu tiên:
  1) tăng chất lượng data (thêm đoạn bắt đầu/kết thúc động tác)
  2) train với pooling `last` hoặc `attn`
  3) kiểm tra threshold/gating trong inference
