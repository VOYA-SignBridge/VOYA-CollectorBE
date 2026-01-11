from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from app.config import settings
from app.routers import dataset, upload, jobs, classes, inference
from app.logging_config import configure_logging
from app.db import init_db

app = FastAPI(title="Sign Dataset Backend")

# Enable CORS for local dev (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# init DB tables (dev). In prod, use migrations (alembic).
@app.on_event("startup")
def startup():
    configure_logging()
    logger = logging.getLogger("startup")
    logger.setLevel(logging.INFO)
    logger.info(f"[CONFIG] dataset_root={settings.dataset_root}")
    init_db()

app.include_router(dataset.router)
app.include_router(upload.router)
app.include_router(jobs.router)
app.include_router(classes.router)
app.include_router(inference.router)