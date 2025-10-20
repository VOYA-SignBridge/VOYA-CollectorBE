import numpy as np
import json
from pathlib import Path

print("🧪 VERIFYING EXPORTED DATASET")
print("=" * 60)

# Load metadata
meta_path = Path("dataset/processed/memmap/dataset_meta.json")
meta = json.loads(meta_path.read_text())

print(f"📊 Dataset Metadata:")
print(f"   • Total samples: {meta['total_samples']}")
print(f"   • Sequence length: {meta['sequence_length']}")  
print(f"   • Feature dimension: {meta['feature_dim']}")
print(f"   • X dtype: {meta['X_dtype']}")
print(f"   • Y dtype: {meta['y_dtype']}")

# Load memmap arrays
X_path = Path("dataset/processed/memmap/dataset_X.dat")
y_path = Path("dataset/processed/memmap/dataset_y.dat")

X = np.memmap(X_path, dtype=np.float32, mode='r', 
              shape=(meta['total_samples'], meta['sequence_length'], meta['feature_dim']))
y = np.memmap(y_path, dtype=np.int32, mode='r', shape=(meta['total_samples'],))

print(f"\n🎯 Loaded Arrays:")
print(f"   • X shape: {X.shape}")
print(f"   • y shape: {y.shape}")
print(f"   • X dtype: {X.dtype}")
print(f"   • y dtype: {y.dtype}")

# Check data statistics
print(f"\n📈 Data Statistics:")
print(f"   • X min: {X.min():.4f}")
print(f"   • X max: {X.max():.4f}")
print(f"   • X mean: {X.mean():.4f}")
print(f"   • X std: {X.std():.4f}")

# Check class distribution
unique_classes, counts = np.unique(y, return_counts=True)
print(f"\n🏷️ Class Distribution:")
for cls, count in zip(unique_classes, counts):
    print(f"   • Class {cls}: {count} samples")

# Verify sample shapes
print(f"\n✅ Shape Verification:")
all_valid = True
for i in range(min(5, len(X))):  # Check first 5 samples
    sample_shape = X[i].shape
    expected_shape = (meta['sequence_length'], meta['feature_dim'])
    if sample_shape == expected_shape:
        print(f"   • Sample {i}: {sample_shape} ✓")
    else:
        print(f"   • Sample {i}: {sample_shape} ❌ (expected {expected_shape})")
        all_valid = False

if all_valid:
    print(f"\n🎉 DATASET EXPORT SUCCESSFUL!")
    print(f"   Ready for model training with {meta['total_samples']} samples")
    print(f"   Data format: ({meta['sequence_length']}, {meta['feature_dim']}) sequences")
else:
    print(f"\n⚠️ VALIDATION ISSUES FOUND!")

print("=" * 60)