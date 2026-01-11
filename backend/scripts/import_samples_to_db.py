"""Import existing samples from dataset/samples/samples.csv into Postgres and optionally upload missing files to MinIO.

Usage:
  python scripts/import_samples_to_db.py --upload-missing
"""
import csv
import argparse
from pathlib import Path
from app.config import settings
from app.storage.metadata_db import ensure_tables, insert_sample
from app.dataset_samples import SAMPLES_CSV
from app.storage.minio_client import upload_file, compute_sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--upload-missing', action='store_true')
    args = parser.parse_args()

    ensure_tables()

    with open(SAMPLES_CSV, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # compute checksum if local file exists
            fpath = Path(row.get('file_path') or '')
            storage_url = row.get('file_path')
            checksum = None
            if fpath.exists() and args.upload_missing and settings.minio_endpoint:
                key = f"features/{getattr(settings,'dataset_version','v0')}/{row.get('language')}/{row.get('dialect')}/{row.get('class_uid')}/{fpath.name}"
                url = upload_file(str(fpath), key)
                if url:
                    storage_url = url
                    checksum = compute_sha256(str(fpath))
            insert_row = {
                'sample_uid': row.get('sample_uid'),
                'class_uid': row.get('class_uid'),
                'slug': row.get('slug'),
                'label_original': row.get('label_original'),
                'language': row.get('language'),
                'dialect': row.get('dialect'),
                'source_type': row.get('source_type'),
                'user_id': row.get('user_id'),
                'session_id': row.get('session_id'),
                'fps_original': row.get('fps_original'),
                'fps_processed': row.get('fps_processed'),
                'seq_len': int(row.get('seq_len') or 0),
                'augment_id': int(row.get('augment_id') or 0),
                'completeness': float(row.get('completeness') or 0.0),
                'file_path': str(row.get('file_path')),
                'storage_url': storage_url,
                'checksum': checksum,
                'created_at': row.get('created_at')
            }
            insert_sample(insert_row)

    print('Import complete')


if __name__ == '__main__':
    main()
