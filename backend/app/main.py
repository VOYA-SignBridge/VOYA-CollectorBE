from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import dataset, upload, jobs
from app.db import init_db

app = FastAPI(title="Sign Dataset Backend")

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# init DB tables (dev). In prod, use migrations (alembic).
@app.on_event("startup")
def startup():
    init_db()

app.include_router(dataset.router)
app.include_router(upload.router)
app.include_router(jobs.router)