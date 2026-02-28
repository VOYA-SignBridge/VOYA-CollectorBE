"""
class_registry.py
UUID-based class registry with multilingual and multi-dialect support.
Handles hierarchical class storage: global_common, language/common, language/dialect.
"""

import os
import csv
import uuid
import unicodedata
import re
import json
import logging
from typing import Optional, Dict, List, Tuple
from datetime import datetime
from filelock import FileLock
from pathlib import Path

logger = logging.getLogger(__name__)

# ---- Config ----
DATASET_ROOT = Path("dataset")
FEATURE_ROOT = DATASET_ROOT / "features"
LABELS_DIR = DATASET_ROOT / "labels"
# Master labels CSV is at dataset root: dataset/labels.csv
LABELS_MASTER_CSV = DATASET_ROOT / "labels.csv"

# ---- Utils ----
def slugify(text: str, maxlen: int = 30) -> str:
    """Convert text (possibly with diacritics) to safe ASCII slug."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join([c for c in text if not unicodedata.combining(c)])
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s_-]", "", text)
    text = re.sub(r"[\s-]+", "_", text).strip("_")
    if len(text) > maxlen:
        text = text[:maxlen].rstrip("_")
    return text if text else "unknown"


def now_str() -> str:
    return datetime.utcnow().isoformat() + "Z"


class ClassInfo:
    """Immutable class metadata"""
    def __init__(
        self,
        class_uid: str,
        slug: str,
        label_original: str,
        language: str,
        dialect: str,
        is_common_global: bool = False,
        is_common_language: bool = False,
        created_at: Optional[str] = None,
        notes: str = "",
    ):
        self.class_uid = class_uid
        self.slug = slug
        self.label_original = label_original
        self.language = language  # 'vn', 'en', 'jp', etc.
        self.dialect = dialect    # 'bac', 'nam', 'common', etc.
        self.is_common_global = is_common_global
        self.is_common_language = is_common_language
        self.created_at = created_at or now_str()
        self.notes = notes

    def to_dict(self) -> Dict:
        return {
            "class_uid": self.class_uid,
            "slug": self.slug,
            "label_original": self.label_original,
            "language": self.language,
            "dialect": self.dialect,
            "is_common_global": str(self.is_common_global),
            "is_common_language": str(self.is_common_language),
            "created_at": self.created_at,
            "notes": self.notes,
        }

    @staticmethod
    def from_dict(data: Dict) -> "ClassInfo":
        return ClassInfo(
            class_uid=data["class_uid"],
            slug=data["slug"],
            label_original=data["label_original"],
            language=data["language"],
            dialect=data["dialect"],
            is_common_global=data.get("is_common_global", "False").lower() == "true",
            is_common_language=data.get("is_common_language", "False").lower() == "true",
            created_at=data.get("created_at"),
            notes=data.get("notes", ""),
        )

    def get_folder_path(self) -> Path:
        """
        Resolve hierarchical folder path based on class properties.
        global_common/<uid>_<slug>
        <lang>/common/<uid>_<slug>
        <lang>/<dialect>/<uid>_<slug>
        """
        folder_name = f"{self.class_uid}_{self.slug}"
        
        if self.is_common_global:
            return FEATURE_ROOT / "global_common" / folder_name
        elif self.is_common_language:
            return FEATURE_ROOT / self.language / "common" / folder_name
        else:
            return FEATURE_ROOT / self.language / self.dialect / folder_name


class ClassRegistry:
    """Thread-safe class registry with UUID-based stable identifiers"""
    
    def __init__(self, labels_csv: Path = LABELS_MASTER_CSV):
        self.labels_csv = labels_csv
        self.lock_path = str(labels_csv) + ".lock"
        self._ensure_schema()

    def _ensure_schema(self):
        """Create labels directory and CSV with headers if missing"""
        os.makedirs(self.labels_csv.parent, exist_ok=True)
        if not self.labels_csv.exists():
            fieldnames = [
                "class_uid", "slug", "label_original", "language", "dialect",
                "is_common_global", "is_common_language", "created_at", "notes"
            ]
            with open(self.labels_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

    def _read_all(self) -> List[ClassInfo]:
        """Read all classes from CSV"""
        if not self.labels_csv.exists():
            return []
        with open(self.labels_csv, newline="", encoding="utf-8") as f:
            return [ClassInfo.from_dict(row) for row in csv.DictReader(f)]

    def _write_all(self, classes: List[ClassInfo]):
        """Write all classes atomically"""
        import tempfile
        lock = FileLock(self.lock_path)
        with lock:
            fd, tmp_path = tempfile.mkstemp(prefix="labels_", suffix=".csv", dir=self.labels_csv.parent)
            os.close(fd)
            try:
                fieldnames = [
                    "class_uid", "slug", "label_original", "language", "dialect",
                    "is_common_global", "is_common_language", "created_at", "notes"
                ]
                with open(tmp_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for cls in classes:
                        writer.writerow(cls.to_dict())
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, self.labels_csv)
            finally:
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except:
                        pass

    def find_class(
        self,
        label_original: Optional[str] = None,
        language: Optional[str] = None,
        dialect: Optional[str] = None,
        slug: Optional[str] = None,
        class_uid: Optional[str] = None,
    ) -> Optional[ClassInfo]:
        """
        Find class by various criteria.
        Priority: class_uid > (language, dialect, slug) > (language, dialect, label_original)
        """
        classes = self._read_all()
        
        if class_uid:
            for cls in classes:
                if cls.class_uid == class_uid:
                    return cls
            return None
        
        if language and slug:
            for cls in classes:
                if cls.language == language and cls.slug == slug:
                    if dialect is None or cls.dialect == dialect:
                        return cls
        
        if language and dialect and label_original:
            for cls in classes:
                if (cls.language == language and 
                    cls.dialect == dialect and 
                    cls.label_original == label_original):
                    return cls
        
        return None

    def register_class(
        self,
        label_original: str,
        language: str,
        dialect: str = "common",
        is_common_global: bool = False,
        is_common_language: bool = False,
        notes: str = "",
        force_new: bool = False,
    ) -> ClassInfo:
        """
        Register a new class or return existing.
        If force_new=True, always create a new class_uid even if similar exists.
        """
        slug = slugify(label_original)
        
        # Check if already exists
        if not force_new:
            existing = self.find_class(
                language=language,
                dialect=dialect,
                slug=slug,
            )
            if existing:
                logger.info(f"Class already exists: {existing.class_uid} ({existing.label_original})")
                return existing
        
        # Create new
        new_uid = str(uuid.uuid4())
        new_class = ClassInfo(
            class_uid=new_uid,
            slug=slug,
            label_original=label_original,
            language=language,
            dialect=dialect,
            is_common_global=is_common_global,
            is_common_language=is_common_language,
            notes=notes,
        )
        
        # Append to CSV
        lock = FileLock(self.lock_path)
        with lock:
            classes = self._read_all()
            classes.append(new_class)
            self._write_all(classes)
        
        # Create folder + metadata.json
        folder = new_class.get_folder_path()
        os.makedirs(folder, exist_ok=True)
        
        metadata_path = folder / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(new_class.to_dict(), f, ensure_ascii=False, indent=2)
        
        logger.info(f"Registered new class: {new_uid} ({label_original}) at {folder}")
        return new_class

    def list_classes(
        self,
        language: Optional[str] = None,
        dialect: Optional[str] = None,
        include_common_global: bool = True,
        include_common_language: bool = True,
    ) -> List[ClassInfo]:
        """
        List classes for a specific language/dialect, optionally including common groups.
        Used for inference-time loading.
        """
        classes = self._read_all()
        result = []
        
        for cls in classes:
            # Include global common
            if include_common_global and cls.is_common_global:
                result.append(cls)
                continue
            
            # Filter by language
            if language and cls.language != language:
                continue
            
            # Include language-level common
            if include_common_language and cls.is_common_language:
                result.append(cls)
                continue
            
            # Filter by dialect
            if dialect and cls.dialect != dialect:
                continue
            
            result.append(cls)
        
        return result

    def get_all_languages(self) -> List[str]:
        """Get all unique languages in registry"""
        classes = self._read_all()
        return sorted(set(cls.language for cls in classes))

    def get_dialects_for_language(self, language: str) -> List[str]:
        """Get all dialects for a specific language"""
        classes = self._read_all()
        return sorted(set(
            cls.dialect for cls in classes 
            if cls.language == language and not cls.is_common_language
        ))
