from typing import Optional, Dict, Any
import os
import urllib.parse
import psycopg2
from psycopg2.extras import RealDictCursor
from app.config import settings


def _get_conn():
    dburl = settings.database_url
    return psycopg2.connect(dburl)


def ensure_tables():
    sql = """
    CREATE TABLE IF NOT EXISTS samples (
        sample_uid TEXT PRIMARY KEY,
        class_uid TEXT,
        slug TEXT,
        label_original TEXT,
        language TEXT,
        dialect TEXT,
        source_type TEXT,
        user_id TEXT,
        session_id TEXT,
        fps_original TEXT,
        fps_processed TEXT,
        seq_len INTEGER,
        augment_id INTEGER,
        completeness REAL,
        file_path TEXT,
        storage_url TEXT,
        checksum TEXT,
        created_at TIMESTAMP WITH TIME ZONE
    );
    """
    conn = _get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(sql)
    finally:
        conn.close()


def insert_sample(row: Dict[str, Any]):
    """Insert a sample row into samples table. Expects keys matching dataset_samples.SAMPLE_FIELDS plus storage_url/checksum."""
    conn = _get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO samples(sample_uid,class_uid,slug,label_original,language,dialect,source_type,user_id,session_id,fps_original,fps_processed,seq_len,augment_id,completeness,file_path,storage_url,checksum,created_at)
                    VALUES(%(sample_uid)s,%(class_uid)s,%(slug)s,%(label_original)s,%(language)s,%(dialect)s,%(source_type)s,%(user_id)s,%(session_id)s,%(fps_original)s,%(fps_processed)s,%(seq_len)s,%(augment_id)s,%(completeness)s,%(file_path)s,%(storage_url)s,%(checksum)s,%(created_at)s)
                    ON CONFLICT (sample_uid) DO NOTHING
                    """,
                    row
                )
    finally:
        conn.close()
