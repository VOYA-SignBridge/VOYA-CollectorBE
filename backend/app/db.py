from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, DateTime, JSON
from sqlalchemy.sql import func
from app.config import settings

engine = create_engine(settings.database_url, pool_pre_ping=True)
metadata = MetaData()

labels = Table(
    "labels", metadata,
    Column("id", Integer, primary_key=True),
    Column("class_idx", Integer, nullable=False),
    Column("label_name", String, nullable=False),
    Column("folder_name", String, nullable=False),
    Column("created_at", DateTime, server_default=func.now())
)

samples = Table(
    "samples", metadata,
    Column("id", Integer, primary_key=True),
    Column("label_id", Integer),
    Column("file_path", String, nullable=False),
    Column("user", String),
    Column("session_id", String),
    Column("frames", Integer),
    Column("duration", String),
    Column("meta", JSON),
    Column("created_at", DateTime, server_default=func.now())
)

def init_db():
    metadata.create_all(engine)
