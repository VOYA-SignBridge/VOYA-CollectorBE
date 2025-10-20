import os, numpy as np
from app.config import settings

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_json_to_storage(obj, path):
    import json
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)

def save_npz_feature(sequence_array, label_folder, filename, meta=None):
    ensure_dir(label_folder)
    outpath = os.path.join(label_folder, filename)
    np.savez_compressed(outpath, sequence=sequence_array.astype(np.float32), meta=meta or {})
    return outpath
