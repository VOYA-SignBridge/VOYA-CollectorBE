import os, uuid
import numpy as np
from app.processing.utils import save_npz_feature

def write_feature_files(sequence_array: np.ndarray, label: str, out_root: str, metadata: dict = None):
    """
    Writes compressed npz to out_root/label_folder/<uuid>.npz
    returns filepath
    """
    label_folder = os.path.join(out_root, label)
    os.makedirs(label_folder, exist_ok=True)
    fname = f"{label}_{uuid.uuid4().hex[:8]}.npz"
    outpath = save_npz_feature(sequence_array, label_folder, fname, meta=metadata)
    return outpath
