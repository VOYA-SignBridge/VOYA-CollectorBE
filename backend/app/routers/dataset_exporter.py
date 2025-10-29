from fastapi import APIRouter, HTTPException
from pathlib import Path
from ..processing.utils import load_npz_features, merge_memmap
from ..processing import validator
from fastapi import Query

router = APIRouter(prefix="/api/dataset", tags=["Dataset Exporter"])

BASE_DATASET_DIR = Path("dataset/features")
OUTPUT_DIR = Path("dataset/processed/memmap")


@router.post("/export")
def export_dataset(fix: bool = Query(False, description="Attempt to auto-fix mismatched samples (pad/truncate) before export")):
    """Aggregate all processed .npz files into unified memmap dataset"""
    try:
        # Validate samples first
        report = validator.validate_samples(BASE_DATASET_DIR, expected_T=60, expected_D=226, fix=fix)
        if not report.get('ok'):
            # If not ok and not fixed, return the report so caller can inspect
            if report.get('fixed_count', 0) == 0:
                raise HTTPException(status_code=400, detail={"message": "Validation failed", "report": report})
        # Reload samples after validation/fix
        samples = load_npz_features(BASE_DATASET_DIR)
        if not samples:
            raise HTTPException(status_code=404, detail="No valid samples found.")
        meta = merge_memmap(samples, OUTPUT_DIR)
        return {
            "status": "success",
            "message": f"Exported {meta['total_samples']} samples.",
            "validation_report": report,
            "output": meta
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
