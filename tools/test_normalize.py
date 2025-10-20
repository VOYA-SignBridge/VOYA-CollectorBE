import sys
import os
import numpy as np

# ensure package imports resolve when running from workspace root
here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# add workspace root and backend folder so imports like `app.processing` resolve
sys.path.insert(0, here)
sys.path.insert(0, os.path.join(here, 'backend'))
from app.processing.pipeline import normalize_sequence

# adjust path to an existing sample in this workspace
# find first .npz in dataset/features
root = os.path.join('dataset', 'features')
sample_path = None
for dirpath, _, files in os.walk(root):
    for f in files:
        if f.endswith('.npz'):
            sample_path = os.path.join(dirpath, f)
            break
    if sample_path:
        break
if sample_path is None:
    raise SystemExit('No .npz samples found under dataset/features')
print('Loading', sample_path)
arr = np.load(sample_path)
seq = arr.get('sequence')
if seq is None:
    # older npz might store named differently
    # try to load the only array
    keys = list(arr.keys())
    if keys:
        seq = arr[keys[0]]

print('Original shape:', None if seq is None else seq.shape)
if seq is None:
    raise SystemExit('No sequence found in npz')

norm = normalize_sequence(seq)
print('Normalized shape:', norm.shape)
# quick sanity checks
print('Pose x mean (first frame):', norm[0, :100].reshape(25,4)[:,:2].mean())
# Save normalized sample for inspection
out_path = 'tools/test_norm.npy'
np.save(out_path, norm)
print('Saved normalized sequence to', out_path)
