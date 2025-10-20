from fastapi import APIRouter, UploadFile, File, Form
from pydantic import BaseModel
from typing import List
import numpy as np
import os
import shutil

from app.processing import storage_utils as su

router = APIRouter(prefix="/dataset", tags=["dataset"])

# ---- Models ----
class LabelOut(BaseModel):
    class_idx: int
    label_original: str
    slug: str
    folder_name: str
    created_at: str
    dataset_version: str
    notes: str

class SampleOut(BaseModel):
    sample_id: str
    class_idx: int
    folder_name: str
    file: str
    user: str
    session_id: str
    frames: str
    duration: str
    source: str
    created_at: str


# ---- Endpoints ----

@router.post("/labels", response_model=LabelOut)
def create_label(label: str = Form(...), notes: str = Form(""), version: str = Form("v1")):
    class_idx, folder = su.register_label(label, notes=notes, dataset_version=version)
    labels = su.read_csv(su.LABELS_CSV)
    new_label = next(r for r in labels if int(r["class_idx"]) == class_idx)
    return new_label


@router.get("/labels", response_model=List[LabelOut])
def list_labels():
    return su.read_csv(su.LABELS_CSV)


@router.post("/labels/merge")
def merge_labels(src_class_idx: int = Form(...), dst_class_idx: int = Form(...)):
    ok = su.merge_labels(src_class_idx, dst_class_idx)
    return {"status": "success" if ok else "failed"}


@router.get("/samples", response_model=List[SampleOut])
def list_samples():
    return su.read_csv(su.SAMPLES_CSV)

@router.get("/samples/{sample_id}/data")
def get_sample_data(sample_id: str):
    """
    Trả về file npz/json của sample_id
    """
    base_dir = "dataset/features"
    # giả sử trong samples.csv có file_path, bạn sẽ lookup từ đó
    import csv
    samples_file = os.path.join(base_dir, "samples.csv")

    with open(samples_file, newline='', encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["sample_id"] == sample_id:
                file_path = row["file_path"]
                return FileResponse(file_path, media_type="application/octet-stream")

    return {"error": "Sample not found"}

@router.post("/samples/add")
def add_sample(
    class_idx: int = Form(...),
    user: str = Form(""),
    session_id: str = Form(""),
    frames: int = Form(0),
    duration: float = Form(0.0),
    source: str = Form("video"),
    file: UploadFile = File(...)
):
    # tìm folder theo class_idx
    labels = su.read_csv(su.LABELS_CSV)
    label = next((r for r in labels if int(r["class_idx"]) == class_idx), None)
    if not label:
        return {"status": "failed", "reason": "label not found"}
    folder = label["folder_name"]

    # lưu file upload tạm thời
    tmp_path = f"/tmp/{file.filename}"
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # đọc npz nếu cần, hoặc giả định file đã là npz chuẩn
    if file.filename.endswith(".npz"):
        seq = np.load(tmp_path)["sequence"]
    else:
        # fallback: random (chỉ demo)
        seq = np.random.rand(60, 1605)

    os.remove(tmp_path)

    # save sample
    metadata = {"user": user, "session_id": session_id, "frames": frames, "duration": duration, "source": source}
    path = su.save_sample(seq, class_idx, folder, metadata=metadata)
    return {"status": "ok", "path": path}

@router.get("/dataset/sessions")
def list_sessions(user: str = "", label: str = "", date: str = ""):
    # Đọc từ CSV hoặc database
    df = pd.read_csv(DATASET_PATH / "samples.csv")

    if user:
        df = df[df["user"] == user]
    if label:
        df = df[df["label"].str.contains(label, case=False)]
    if date:
        df = df[df["date"].str.startswith(date)]

    sessions = []
    for sid, group in df.groupby("session_id"):
        sessions.append({
            "session_id": sid,
            "user": group["user"].iloc[0],
            "labels": list(group["label"].unique()),
            "samples_count": len(group),
            "created_at": group["date"].iloc[0],
        })

    return sessions