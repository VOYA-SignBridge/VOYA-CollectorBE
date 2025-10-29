from pathlib import Path
import numpy as np
import json
import shutil

# Working dir assumed to be repository backend root
BASE = Path(".")
FEATURE_DIR = BASE / "dataset" / "features" / "class_0001_test"
proc_dir = BASE / "dataset" / "processed" / "memmap"

# cleanup previous
if FEATURE_DIR.exists():
    shutil.rmtree(FEATURE_DIR)
FEATURE_DIR.mkdir(parents=True, exist_ok=True)
if proc_dir.exists():
    shutil.rmtree(proc_dir)

# create two small npz samples
for i in range(2):
    seq = np.random.randn(60, 226).astype(np.float32)
    sample_name = FEATURE_DIR / f"sample_000{i+1}.npz"
    np.savez_compressed(sample_name, sequence=seq)
    meta = {"class_idx": 1, "created_at": "test"}
    meta_path = sample_name.with_suffix('.json')
    meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding='utf-8')

print("Created test samples:")
for p in FEATURE_DIR.glob('*.npz'):
    print(' -', p)

# import utils and run
from app.processing.utils import load_npz_features, merge_memmap

samples = load_npz_features(FEATURE_DIR)
print('Loaded', len(samples), 'samples')
meta = merge_memmap(samples, proc_dir)
print('merge result:')
print(meta)

# verify memmap file exists
m = Path(meta['memmap_path'])
print('memmap exists:', m.exists())
# load via memmap
import numpy as np
mm = np.memmap(str(m), dtype='float32', mode='r', shape=tuple(meta['shape']))
print('memmap shape:', mm.shape)

print('Test export completed')
