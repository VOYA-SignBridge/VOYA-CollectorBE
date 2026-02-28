from typing import Optional
import hashlib
import os
from minio import Minio
from minio.error import S3Error
from app.config import settings


def _get_minio_client() -> Optional[Minio]:
    if not settings.minio_endpoint or not settings.minio_access_key or not settings.minio_secret_key:
        return None
    secure = settings.minio_endpoint.startswith("https://")
    endpoint = settings.minio_endpoint.replace("https://", "").replace("http://", "")
    return Minio(endpoint, access_key=settings.minio_access_key, secret_key=settings.minio_secret_key, secure=secure)


def compute_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_bucket(client: Minio, bucket: str):
    found = client.bucket_exists(bucket)
    if not found:
        client.make_bucket(bucket)


def upload_file(path: str, key: str) -> Optional[str]:
    """Upload local file at `path` to configured MinIO with object key `key`.
    Returns storage URL on success or None on failure.
    """
    client = _get_minio_client()
    if client is None:
        return None
    bucket = settings.minio_bucket or "sign-dataset"
    try:
        ensure_bucket(client, bucket)
        client.fput_object(bucket, key, path)
        endpoint = settings.minio_endpoint.rstrip("/")
        # Basic public URL; if MinIO is behind proxy this may vary
        url = f"{endpoint}/{bucket}/{key}"
        return url
    except S3Error:
        return None
