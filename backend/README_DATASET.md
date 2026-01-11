# Multilingual Sign Dataset Architecture

## Overview
This backend manages a hierarchical, multilingual, multi-dialect sign language dataset optimized for TCN training (sequence length=60, feature_dim=126).

## Directory Structure
```
dataset/
  raw_videos/
  raw_live/
  features/
    global_common/<class_uid>_<slug>/
    <language>/
      common/<class_uid>_<slug>/
      <dialect>/<class_uid>_<slug>/
  labels/
    labels_master.csv
    labels_language.csv
    labels_dialect.csv
  samples/
    samples.csv
```
Each class folder contains `metadata.json`, sample `.npz` files (with embedded meta + json sidecars).

## Class Metadata
```
{
  "class_uid": "uuid4",
  "slug": "cam_on",
  "label_original": "cảm ơn",
  "language": "vn",
  "dialect": "bac" | "nam" | "common" | "global",
  "is_common_global": true/false,
  "is_common_language": true/false
}
```

## Sample Metadata (samples.csv)
Fields:
```
sample_uid,class_uid,slug,label_original,language,dialect,source_type,user_id,session_id,fps_original,fps_processed,seq_len,augment_id,completeness,file_path,created_at
```

## Pipelines
- Video: streaming decode, optional ffmpeg resample to 30 FPS, MediaPipe Hands only → 126D per frame. Sliding window (60, stride configurable default=2), carry-forward missing frames (setting `CARRY_FORWARD_MISSING=1`). Augmentation parity with live capture.
- Live: direct frame landmark payload → normalization + augmentation.

## Balancing
- `balancer.py` computes target_per_class and plan.
- `oversample_balance.py` executes augmentation plan.
- Endpoint `/classes/balance` returns plan JSON.

## Inference Loading
Use `inference_loader.py`:
```python
from app.inference_loader import load_inference_classes, build_class_map
classes = load_inference_classes(language="vn", dialect="bac")
class_map = build_class_map(classes)
```
Returns selected dialect + language common + global common.

## Promotion to Global Common
`global_common_promoter.py` identifies slugs appearing across multiple languages and can promote them:
```bash
python -m app.global_common_promoter --min-languages 2 --dry-run
python -m app.global_common_promoter --min-languages 2 --promote
```

## Validation
Run validator:
```bash
docker-compose exec backend python -m app.validation.dataset_validator --all
```
Outputs completeness stats, imbalance ratios, dialect confusion, cross-language collisions.

## Migration
```
python -m app.migrations.migrate_flat_to_hierarchy --language vn --language-common
```
Re-saves legacy flat samples.

After verifying new hierarchy, list legacy folders:
```
python -m app.feature_structure_audit
```
Remove legacy `class_XXXX_*` roots once safe:
```
python -m app.cleanup_legacy_features --delete
```

## Configuration Flags
ENV vars:
- `STRIDE` (default 2)
- `SEQ_LEN` (default 60)
- `FEATURE_DIM` (default 126)
- `MAX_SAMPLES_PER_CLASS` (default 80; global cap per class across the whole dataset; video ingestion stops saving once the class reaches this cap)
- `AUG_PER_SEQ` (default 8)
- `VIDEO_AUG_PER_SEQ` (default = `AUG_PER_SEQ`; set higher to reach the per-class cap faster from short videos)
- `VIDEO_MIN_SAMPLES_PER_VIDEO` (default 10; best-effort minimum samples saved per video upload, capped by remaining `MAX_SAMPLES_PER_CLASS`; set 0 to disable)
- `MIRROR_INPUT` (default 0; set 1 if camera preview is mirrored and you want to un-mirror before extracting keypoints)
- `NORMALIZE_KEYPOINTS` (default 0; set 1 to center+scale normalize hand vectors)
- `VIDEO_SKIP_LEADING_NO_HAND` (default 1; skip leading frames without detected hands when ingesting videos)
- `VIDEO_ACTIVITY_THRESHOLD` (default 0.0 disabled; reject windows with low motion even if hands are present)
- `CANONICALIZE_HANDS` (default 1; canonicalize hand vectors so left-vs-right hand usage is more consistent)
- `CANONICALIZE_MIRROR` (default 1; choose canonical orientation between original and mirrored+swapped vectors)
- `DATASET_VERSION` (default v0; used in MinIO object keys)
- `DATASET_UPLOAD_REMOVE_LOCAL` (default 0; set 1 to delete local npz after successful MinIO upload)
- `CARRY_FORWARD_MISSING` (1 enable carry-forward for missing hand frames)
- `DEBUG_LOGGING` (1 to enable debug output)

## Endpoints Summary
- `POST /classes/register` register new class.
- `GET /classes/list?language=&dialect=` list classes.
- `GET /classes/stats` distribution and imbalance ratios.
- `GET /classes/balance` augmentation plan.
- `GET /feature/audit` (script only) for legacy structure reporting.
- `GET /inference/classes?language=vn&dialect=bac` inference mapping.
- `POST /upload/video` video ingestion.
- `POST /upload/camera` live capture ingestion.

## Future Extensions
- Memmap exporter for faster training.
- Automatic semantic similarity promotion to global_common.
- Dataset splits manifest generation.

## Backward Compatibility
Legacy flat structure remains until you run cleanup. Migration preserves originals; cleanup script deletes them once confirmed.

## Logging
Set `DEBUG_LOGGING=1` to view frame-level and resample debug logs.

---
This document reflects current implementation state for multilingual scalable sign dataset management.
