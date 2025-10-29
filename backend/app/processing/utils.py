import os, numpy as np
from app.config import settings
from pathlib import Path
import json

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


def load_npz_features(base_dir: Path):
    """Load all .npz feature files under base_dir.

    Returns a list of dicts: { 'sequence': ndarray(T,D), 'class_idx': int|None, 'path': Path, 'meta': dict }
    """
    base = Path(base_dir)
    files = list(base.rglob("*.npz"))
    samples = []
    for p in files:
        try:
            data = np.load(p, allow_pickle=False)
        except Exception:
            # allow fallback to pickle for meta if needed (sequence should still load)
            data = np.load(p, allow_pickle=True)
        seq = None
        if 'sequence' in data:
            seq = data['sequence']
        elif 'sequences' in data:
            seq = data['sequences']
        else:
            # skip files without sequence
            continue

        # prefer external json metadata if present
        meta = {}
        meta_path = p.with_suffix('.json')
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding='utf-8'))
            except Exception:
                meta = {}
        else:
            # try to extract 'meta' inside npz if available
            try:
                if 'meta' in data:
                    # meta may be stored as an array/object; attempt to coerce to dict
                    raw = data['meta']
                    # if it's an array-like object with item(), use that
                    try:
                        meta = raw.item()
                    except Exception:
                        try:
                            meta = dict(raw)
                        except Exception:
                            meta = {}
            except Exception:
                meta = {}

        class_idx = None
        try:
            class_idx = int(meta.get('class_idx')) if meta.get('class_idx') is not None else None
        except Exception:
            class_idx = None

        samples.append({
            'sequence': np.asarray(seq, dtype=np.float32),
            'class_idx': class_idx,
            'path': p,
            'meta': meta,
        })

    # sort samples by path for deterministic order
    samples.sort(key=lambda s: str(s['path']))
    return samples


def merge_memmap(samples, output_dir: Path):
    """Merge loaded samples into a single numpy.memmap file on disk.

    samples: list from load_npz_features
    output_dir: Path where memmap and metadata will be written

    Returns meta dict with keys: total_samples, shape, dtype, memmap_path, meta_path
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not samples:
        raise ValueError("No samples provided to merge_memmap")

    # infer shape from first sample
    first = samples[0]['sequence']
    if first.ndim != 2:
        raise ValueError("Expected 2D sequences (T,D)")
    T, D = first.shape
    N = len(samples)

    # verify compatible shapes
    for s in samples:
        seq = s['sequence']
        if seq.ndim != 2:
            raise ValueError(f"Sample {s['path']} has ndim!=2")
        if seq.shape[1] != D:
            raise ValueError(f"Feature dim mismatch for {s['path']}: {seq.shape[1]} != {D}")
        if seq.shape[0] != T:
            # allow sequences with T different (should be fixed by validator) but truncate/pad if needed
            # simple behavior: truncate or pad with zeros
            arr = np.zeros((T, D), dtype=np.float32)
            if seq.shape[0] >= T:
                arr[:] = seq[:T, :]
            else:
                arr[:seq.shape[0], :] = seq
            s['sequence'] = arr

    memmap_path = out / "features.dat"
    # create memmap file
    mmap = np.memmap(str(memmap_path), dtype='float32', mode='w+', shape=(N, T, D))
    for i, s in enumerate(samples):
        mmap[i, :, :] = s['sequence']

    mmap.flush()

    # write metadata
    meta = {
        'total_samples': N,
        'shape': [N, T, D],
        'dtype': 'float32',
        'memmap_path': str(memmap_path),
    }
    meta_path = out / 'meta.json'
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')

    return {**meta, 'meta_path': str(meta_path)}
