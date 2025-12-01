# app/services/minio_client.py

from minio import Minio
from minio.error import S3Error
from app.core.config import settings
from app.core.logging import logger
import os
from io import BytesIO

class MinioService:
    def __init__(self):
        # Note: MinIO requires secure=False for local HTTP
        self.client = Minio(
            endpoint=settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=False,
        )
        
        # Ensure bucket exists
        for bucket in [settings.MINIO_BUCKET_DOCS, settings.MINIO_BUCKET_PPT_IMAGES]:
            if not self.client.bucket_exists(bucket_name=bucket):
                self.client.make_bucket(bucket_name=bucket)
                logger.info(f"[MinIO] Created bucket: {bucket}")
    
    def upload_file(self, file_path: str, bucket: str, object_name: str = None) -> str:
        """Upload a file to MinIO bucket."""
        if object_name is None:
            object_name = os.path.basename(file_path)
        
        try:
            self.client.fput_object(
                bucket_name=bucket,
                object_name=object_name,
                file_path=file_path,
            )
            logger.info(f"[MinIO] Uploaded {file_path} -> {bucket}/{object_name}")
            return f"{bucket}/{object_name}"
        except S3Error as e:
            logger.error(f"[MinIO] Error uploading {file_path}: {e}")
            raise
    
    def upload_bytes(self, data: bytes, bucket: str, object_name: str, content_type: str = "application/json") -> str:
        """Upload bytes data to MinIO bucket."""
        try:
            self.client.put_object(
                bucket_name=bucket,
                object_name=object_name,
                data=BytesIO(data),
                length=len(data),
                content_type=content_type,
            )
            logger.info(f"[MinIO] Uploaded bytes -> {bucket}/{object_name}")
            return f"{bucket}/{object_name}"
        except S3Error as e:
            logger.error(f"[MinIO] Error uploading bytes to {bucket}/{object_name}: {e}")
            raise
    
    def download_as_text(self, bucket: str, object_name: str) -> str:
        """Download object and return as text string."""
        try:
            response = self.client.get_object(bucket_name=bucket, object_name=object_name)
            text = response.read().decode('utf-8')
            response.close()
            response.release_conn()
            logger.info(f"[MinIO] Downloaded {bucket}/{object_name} as text")
            return text
        except S3Error as e:
            logger.error(f"[MinIO] Error downloading {bucket}/{object_name}: {e}")
            raise
        except UnicodeDecodeError as e:
            logger.error(f"[MinIO] Error decoding {bucket}/{object_name} as UTF-8: {e}")
            raise
    
    def list_files(self, bucket: str, prefix: str = "", suffix: str = "") -> list:
        """List files in bucket matching prefix and suffix."""
        try:
            objects = self.client.list_objects(bucket_name=bucket, prefix=prefix, recursive=True)
            files = []
            for obj in objects:
                if suffix == "" or obj.object_name.endswith(suffix):
                    files.append(obj.object_name)
            logger.info(f"[MinIO] Listed {len(files)} files in {bucket} with prefix='{prefix}' suffix='{suffix}'")
            return files
        except S3Error as e:
            logger.error(f"[MinIO] Error listing files in {bucket}: {e}")
            return []
