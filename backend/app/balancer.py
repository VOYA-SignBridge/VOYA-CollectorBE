"""balancer.py
Compute dataset balance statistics and produce augmentation/oversampling plan.
"""
from __future__ import annotations

from typing import Dict, List
from collections import defaultdict
from app.dataset_samples import list_samples
from app.dataset_manager import load_labels, ClassMetadata

def count_samples_per_class() -> Dict[str, int]:
    samples = list_samples()
    counts: Dict[str, int] = defaultdict(int)
    for s in samples:
        cid = s.get("class_uid")
        if cid:
            counts[cid] += 1
    return counts

def load_class_meta_map() -> Dict[str, ClassMetadata]:
    rows = load_labels()
    out: Dict[str, ClassMetadata] = {}
    for r in rows:
        out[r["class_uid"]] = ClassMetadata(
            class_uid=r["class_uid"],
            slug=r["slug"],
            label_original=r["label_original"],
            language=r["language"],
            dialect=r["dialect"],
            is_common_global=bool(int(r["is_common_global"])),
            is_common_language=bool(int(r["is_common_language"])),
        )
    return out

def build_balance_plan(target: int | None = None, min_threshold: int | None = None) -> Dict:
    counts = count_samples_per_class()
    meta_map = load_class_meta_map()
    if not counts:
        return {"error": "no samples"}
    max_count = max(counts.values())
    target_count = target or max_count
    plan = []
    for cid, current in counts.items():
        if current >= target_count:
            continue
        deficit = target_count - current
        if min_threshold and current >= min_threshold:
            continue
        m = meta_map.get(cid)
        plan.append({
            "class_uid": cid,
            "slug": m.slug if m else "?",
            "label_original": m.label_original if m else "?",
            "language": m.language if m else "?",
            "dialect": m.dialect if m else "?",
            "current": current,
            "target": target_count,
            "deficit": deficit,
            "strategy": "augment_existing",
        })
    return {"target_per_class": target_count, "max_count": max_count, "plan": plan}

if __name__ == "__main__":
    print(build_balance_plan())
