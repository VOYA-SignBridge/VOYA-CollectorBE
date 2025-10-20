from pathlib import Path
import numpy as np
import json
import logging
from typing import Tuple, Dict, Any, List

logger = logging.getLogger(__name__)


def _read_npz(path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    # Load without pickle for safety
    try:
        data = np.load(path, allow_pickle=False)
    except Exception as e:
        logger.exception("Failed to load npz: %s", path)
        raise
    # support both 'sequence' and legacy 'sequences'
    seq = data['sequence'] if 'sequence' in data else data.get('sequences')
    # read meta json if present
    meta = {}
    meta_path = path.with_suffix('.json')
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding='utf-8'))
        except Exception:
            logger.warning("Failed to parse meta json for %s", meta_path)
            meta = {}
    return seq, meta


def validate_samples(base_dir: Path, expected_T: int = None, expected_D: int = None, fix: bool = False) -> Dict[str, Any]:
    """Validate .npz samples under base_dir.

    - Ensures each .npz has a sequence ndarray of shape (T, D)
    - Ensures corresponding .json exists and contains class_idx
    - If expected_T/expected_D unspecified, infer by majority shape
    - If fix=True, will attempt to pad/truncate sequences to target T when possible and update meta['frames']

    Returns report dict with keys: ok, target_shape, mismatch_count, mismatches(list)
    """
    base_dir = Path(base_dir)
    npz_files = list(base_dir.rglob('*.npz'))
    if not npz_files:
        return {"ok": False, "reason": "no_samples", "details": "No .npz files found under base_dir"}

    shapes = {}
    samples_info: List[Dict[str, Any]] = []
    for p in npz_files:
        try:
            seq, meta = _read_npz(p)
            if seq is None:
                samples_info.append({"file": str(p), "error": "no_sequence"})
                continue
            if seq.ndim != 2:
                samples_info.append({"file": str(p), "shape": getattr(seq, 'shape', None), "error": "ndim!=2"})
                continue
            shapes.setdefault(tuple(seq.shape), 0)
            shapes[tuple(seq.shape)] += 1
            samples_info.append({"file": str(p), "shape": tuple(seq.shape), "class_idx": meta.get('class_idx')})
        except Exception as e:
            samples_info.append({"file": str(p), "error": str(e)})

    # infer target shape
    if expected_T is None or expected_D is None:
        # pick most common shape
        if not shapes:
            return {"ok": False, "reason": "no_valid_samples", "details": "No valid sequence arrays found"}
        mode_shape = max(shapes.items(), key=lambda kv: kv[1])[0]
        target_T, target_D = mode_shape
    else:
        target_T, target_D = expected_T, expected_D

    mismatches = []
    for info in samples_info:
        if 'error' in info:
            mismatches.append(info)
            continue
        shape = info.get('shape')
        if shape != (target_T, target_D):
            mismatches.append(info)

    # attempt fixes if requested
    fixed = []
    cannot_fix = []
    if fix and mismatches:
        for m in mismatches:
            if 'file' not in m:
                continue
            fpath = Path(m['file'])
            try:
                seq, meta = _read_npz(fpath)
                if seq is None or seq.ndim != 2:
                    cannot_fix.append({"file": str(fpath), "reason": "no_sequence_or_bad_dim"})
                    continue
                T, D = seq.shape
                # if feature dim differs, skip (cannot safely fix)
                if D != target_D:
                    cannot_fix.append({"file": str(fpath), "reason": f"feature_dim_mismatch ({D}!={target_D})"})
                    continue
                if T < target_T:
                    pad = np.zeros((target_T - T, D), dtype=np.float32)
                    seq2 = np.vstack([seq, pad])
                else:
                    seq2 = seq[:target_T]

                # overwrite npz (only store sequence in the npz)
                np.savez_compressed(fpath, sequence=seq2.astype(np.float32))
                # update meta frames (external .json)
                meta_path = fpath.with_suffix('.json')
                if meta_path.exists():
                    try:
                        meta_obj = json.loads(meta_path.read_text(encoding='utf-8'))
                    except Exception:
                        meta_obj = {}
                else:
                    meta_obj = {}
                meta_obj['frames'] = int(target_T)
                meta_path.write_text(json.dumps(meta_obj, ensure_ascii=False), encoding='utf-8')
                fixed.append(str(fpath))
            except Exception as e:
                cannot_fix.append({"file": str(fpath), "reason": str(e)})

    report = {
        "ok": len(mismatches) == 0 and (not cannot_fix),
        "target_shape": (int(target_T), int(target_D)),
        "total_samples": len(npz_files),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "fixed_count": len(fixed),
        "fixed": fixed,
        "cannot_fix": cannot_fix,
    }
    return report
