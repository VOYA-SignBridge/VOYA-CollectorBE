"""cleanup_legacy_features.py
Optionally remove or relocate legacy flat feature folders (class_XXXX_slug) after migration.
By default lists candidates; use --delete to remove.
"""
from __future__ import annotations
import argparse, shutil
from pathlib import Path
from app.feature_structure_audit import scan

FEATURES_ROOT = Path("dataset") / "features"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delete", action="store_true", help="Delete legacy root class_* folders")
    args = ap.parse_args()
    res = scan()
    legacy = res.get("legacy_roots", [])
    print({"legacy": legacy, "delete": args.delete})
    if args.delete:
        for name in legacy:
            p = FEATURES_ROOT / name
            if p.exists():
                shutil.rmtree(p, ignore_errors=True)
        print({"removed": len(legacy)})

if __name__ == "__main__":
    main()
