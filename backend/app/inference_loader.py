"""inference_loader.py
Load classes for realtime inference filtered by (language, dialect) including common and global_common groups.
"""
from __future__ import annotations

from typing import List, Dict
from app.dataset_manager import load_labels, ClassMetadata

def load_inference_classes(language: str, dialect: str) -> List[ClassMetadata]:
    rows = load_labels()
    out: List[ClassMetadata] = []
    for r in rows:
        lang = r["language"]
        dia = r["dialect"]
        is_global = bool(int(r["is_common_global"]))
        is_lang_common = bool(int(r["is_common_language"]))
        # Include:
        # - Matching language+dialect
        # - Language common for that language
        # - Global common always
        if is_global or (lang == language and (dia == dialect or is_lang_common)):
            out.append(ClassMetadata(
                class_uid=r["class_uid"],
                slug=r["slug"],
                label_original=r["label_original"],
                language=lang,
                dialect=dia,
                is_common_global=is_global,
                is_common_language=is_lang_common,
            ))
    return out

def build_class_map(classes: List[ClassMetadata]) -> Dict[str, Dict[str, str]]:
    return {
        c.class_uid: {
            "slug": c.slug,
            "label_original": c.label_original,
            "language": c.language,
            "dialect": c.dialect,
            "is_common_global": c.is_common_global,
            "is_common_language": c.is_common_language,
        }
        for c in classes
    }
