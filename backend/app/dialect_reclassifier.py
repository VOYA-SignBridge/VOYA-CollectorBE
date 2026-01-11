"""dialect_reclassifier.py (scan & auto-move)
Scan feature hierarchy, compare folder dialect vs metadata.json dialect field,
and produce a move plan. Optionally execute moves.

Expected structure examples:
    dataset/features/<language>/common/<class_uid>_<slug>
    dataset/features/<language>/<dialect>/<class_uid>_<slug>

Decision rules:
    Folder in 'common' but metadata.dialect != 'common'  -> move to that dialect.
    Folder in dialect folder but metadata.dialect == 'common' -> move to 'common'.
    Folder dialect != metadata.dialect and metadata.dialect not in {common, folder} -> move to metadata.dialect.

Usage:
    python -m app.dialect_reclassifier --root D:/path/to/dataset/features --dry-run
    python -m app.dialect_reclassifier --root D:/path/to/dataset/features --apply
"""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
from typing import List, Dict

def iter_class_dirs(features_root: Path) -> List[Path]:
    """Collect class directories in new hierarchy AND legacy flat folders.

    New hierarchy patterns:
      features/<language>/common/<class_uid>_<slug>/
      features/<language>/<dialect>/<class_uid>_<slug>/

    Legacy flat (detected):
      features/<class_uid>_<slug>/ (contains metadata.json) OR features/class_* style
      features/<language>/<class_uid>_<slug>/ (missing dialect/common layer)
    """
    out: List[Path] = []
    if not features_root.exists():
        return out

    # First pass: new hierarchy
    for lang_dir in features_root.iterdir():
        if not lang_dir.is_dir() or lang_dir.name in ("global_common",):
            continue
        for sub in lang_dir.iterdir():
            if not sub.is_dir():
                continue
            if sub.name == "common":
                for cls in sub.iterdir():
                    if cls.is_dir():
                        out.append(cls)
            else:
                # dialect folder
                for cls in sub.iterdir():
                    if cls.is_dir():
                        out.append(cls)

    # Second pass: legacy folders directly under features_root
    for legacy in features_root.iterdir():
        if not legacy.is_dir():
            continue
        if legacy.name in ("global_common",):
            continue
        # Skip language directories already processed above
        # Legacy folders often start with 'class_' or have pattern <uuid>_slug
        meta = legacy / "metadata.json"
        if meta.exists():
            # Ensure not a language dir (language dirs contain subdirs like 'common')
            # Heuristic: if folder contains .npz files directly it's a legacy class dir
            has_npz = any(p.suffix == ".npz" for p in legacy.iterdir() if p.is_file())
            if has_npz:
                out.append(legacy)

    # Third pass: legacy one-level language/<class_dir>
    for lang_dir in features_root.iterdir():
        if not lang_dir.is_dir() or lang_dir.name in ("global_common",):
            continue
        # If this language dir directly contains metadata-bearing folders without dialect/common wrapper
        for maybe_cls in lang_dir.iterdir():
            if not maybe_cls.is_dir():
                continue
            if maybe_cls.name in ("common",):
                continue  # already handled
            # Skip if this is a dialect folder (contains class subfolders with metadata)
            # Heuristic: dialect folder will have subdirectories each with metadata.json
            sub_with_meta = sum(1 for s in maybe_cls.iterdir() if s.is_dir() and (s / "metadata.json").exists()) if any(maybe_cls.iterdir()) else 0
            if sub_with_meta >= 1 and (maybe_cls / "metadata.json").exists() is False:
                continue  # treat as dialect folder already processed above
            if (maybe_cls / "metadata.json").exists():
                out.append(maybe_cls)
    return out

def read_metadata(cls_dir: Path) -> Dict:
    meta_path = cls_dir / "metadata.json"
    if not meta_path.exists():
        return {}
    try:
        import json
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def classify(cls_dir: Path) -> Dict:
    parts = cls_dir.parts
    try:
        idx = parts.index("features")
    except ValueError:
        return {"path": str(cls_dir), "error": "features segment not found"}

    meta = read_metadata(cls_dir)
    md_language = meta.get("language") or ""
    md_dialect = meta.get("dialect")

    # Determine folder language & dialect layer heuristically
    # Cases:
    # 1) New hierarchy depth >= features/<lang>/<dialect|common>/<class>
    # 2) Legacy depth == features/<class>
    # 3) Legacy depth == features/<lang>/<class>
    depth_after_features = len(parts) - (idx + 1)
    language = ""; dialect_folder = "legacy_direct"  # default markers

    if depth_after_features >= 3:
        language = parts[idx+1]
        dialect_folder = parts[idx+2]
    elif depth_after_features == 1:
        # features/<class>
        language = md_language or "unknown"
        dialect_folder = "legacy_flat"
    elif depth_after_features == 2:
        # features/<lang>/<class>
        language = parts[idx+1]
        dialect_folder = "legacy_flat_lang"
    else:
        return {"path": str(cls_dir), "error": "unhandled depth"}

    # Normalize target move logic only for legacy markers or mismatch
    needs = determine_need_move(dialect_folder, md_dialect)
    target = target_dialect(dialect_folder, md_dialect)
    # For legacy folders, compute target from metadata dialect if available
    if dialect_folder.startswith("legacy") and md_dialect:
        if md_dialect == "common":
            target = "common"
            needs = True
        else:
            target = md_dialect
            needs = True

    return {
        "path": str(cls_dir),
        "language": language or (md_language or "unknown"),
        "dialect_folder": dialect_folder,
        "metadata_dialect": md_dialect,
        "class_uid": meta.get("class_uid"),
        "slug": meta.get("slug"),
        "label_original": meta.get("label_original"),
        "needs_move": needs,
        "target_dialect": target,
    }

from typing import Optional

def determine_need_move(dialect_folder: str, md_dialect: Optional[str]) -> bool:
    if md_dialect is None:
        return False
    if dialect_folder == "common" and md_dialect != "common":
        return True
    if dialect_folder != "common" and md_dialect == "common":
        return True
    if dialect_folder != md_dialect and md_dialect not in ("common", dialect_folder):
        return True
    return False

def target_dialect(dialect_folder: str, md_dialect: Optional[str]) -> Optional[str]:
    if md_dialect is None:
        return None
    if dialect_folder == "common" and md_dialect != "common":
        return md_dialect
    if dialect_folder != "common" and md_dialect == "common":
        return "common"
    if dialect_folder != md_dialect and md_dialect not in ("common", dialect_folder):
        return md_dialect
    return None

def apply_moves(features_root: Path, plan: List[Dict]):
    performed = []
    for item in plan:
        if not item.get("needs_move"):
            continue
        target_dialect = item.get("target_dialect")
        if not target_dialect:
            continue
        cls_path = Path(item["path"])
        language = item["language"]
        dest_dir = features_root / language / target_dialect
        dest_dir.mkdir(parents=True, exist_ok=True)
        new_path = dest_dir / cls_path.name
        shutil.move(str(cls_path), str(new_path))
        performed.append({"from": str(cls_path), "to": str(new_path)})
    return performed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Path to features root (e.g. D:/.../dataset/features)")
    ap.add_argument("--apply", action="store_true", help="Execute moves instead of dry run")
    args = ap.parse_args()
    features_root = Path(args.root).expanduser().resolve()
    class_dirs = iter_class_dirs(features_root)
    report = [classify(d) for d in class_dirs]
    plan = [r for r in report if r.get("needs_move")]
    result = {"total": len(report), "needs_move": len(plan), "plan": plan}
    if args.apply and plan:
        performed = apply_moves(features_root, plan)
        result["performed"] = performed
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
