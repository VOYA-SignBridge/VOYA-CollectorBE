from fastapi import APIRouter, Query
from app.inference_loader import load_inference_classes, build_class_map

router = APIRouter(prefix="/inference", tags=["inference"])

@router.get("/classes")
def inference_classes(language: str = Query("vn"), dialect: str = Query("")):
    classes = load_inference_classes(language, dialect)
    return {"language": language, "dialect": dialect, "count": len(classes), "classes": build_class_map(classes)}
