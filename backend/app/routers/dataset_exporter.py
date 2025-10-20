from fastapi import APIRouter, HTTPException
from pathlib import Path
import os
import logging
from ..processing.utils import load_npz_features, merge_memmap
from ..processing import validator
from fastapi import Query

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/dataset", tags=["Dataset Exporter"])

# Make paths configurable via env vars; default to local dataset paths
BASE_DATASET_DIR = Path(os.getenv('SIGN_BASE_DATASET_DIR', 'dataset/features'))
OUTPUT_DIR = Path(os.getenv('SIGN_OUTPUT_DIR', 'dataset/processed/memmap'))


@router.post("/export")
def export_dataset(fix: bool = Query(False, description="Attempt to auto-fix mismatched samples (pad/truncate) before export")):
    """Aggregate all processed .npz files into unified memmap dataset"""
    try:
        logger.info("Export requested: base=%s output=%s fix=%s", BASE_DATASET_DIR, OUTPUT_DIR, fix)
        # Validate samples first
        report = validator.validate_samples(BASE_DATASET_DIR, expected_T=60, expected_D=226, fix=fix)
        if not report.get('ok'):
            # If not ok and not fixed, return the report so caller can inspect
            if report.get('fixed_count', 0) == 0:
                logger.warning("Validation failed: %s", report)
                raise HTTPException(status_code=400, detail={"message": "Validation failed", "report": report})
        # Reload samples after validation/fix
        samples = load_npz_features(BASE_DATASET_DIR)
        if not samples:
            logger.warning("No samples found in %s", BASE_DATASET_DIR)
            raise HTTPException(status_code=404, detail="No valid samples found.")
        meta = merge_memmap(samples, OUTPUT_DIR)
        logger.info("Export completed: %s", meta)
        return {
            "status": "success",
            "message": f"Exported {meta['total_samples']} samples.",
            "validation_report": report,
            "output": meta
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Export failed")
        raise HTTPException(status_code=500, detail=str(e))
