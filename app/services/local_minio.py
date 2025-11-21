import os
from minio import Minio
import mimetypes
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------
# MinIO Client
# ---------------------------------------
client = Minio(
    os.getenv("MINIO_ENDPOINT"),
    access_key=os.getenv("MINIO_ACCESS_KEY"),
    secret_key=os.getenv("MINIO_SECRET_KEY"),
    secure = False
)

# ---------------------------------------
# MAIN ROOT FOLDER
# ---------------------------------------
ROOT_FOLDER = os.getenv("DATA_FOLDER")

# ---------------------------------------
# Folder-to-Bucket Mapping
# ---------------------------------------
FOLDER_BUCKET_MAP = {
    "PPTs": "ppt-bucket",
    "Stocks": "stocks-bucket",
    "Earnings_Release": "earnings-bucket",
    "Transcript": "transcripts-bucket",
    "PPT_Json": "ppt-json-bucket"
}

# ---------------------------------------
# Ensure bucket exists
# ---------------------------------------
def ensure_bucket(bucket_name):
    if not client.bucket_exists(bucket_name):
        client.make_bucket(bucket_name)
        print(f"Bucket created: {bucket_name}")
    else:
        print(f"Bucket exists: {bucket_name}")

# ---------------------------------------
# Upload File
# ---------------------------------------
def upload_file(bucket, file_path, filename):
    content_type, _ = mimetypes.guess_type(file_path)
    if content_type is None:
        content_type = "application/octet-stream"

    client.fput_object(
        bucket,
        filename,
        file_path,
        content_type=content_type
    )
    print(f"Uploaded: {filename} → {bucket}")

# ---------------------------------------
# Main Logic
# ---------------------------------------
for folder_name in os.listdir(ROOT_FOLDER):
    folder_path = os.path.join(ROOT_FOLDER, folder_name)

    # Skip non-folders
    if not os.path.isdir(folder_path):
        continue

    # Skip unknown folders
    if folder_name not in FOLDER_BUCKET_MAP:
        print(f"Skipping unknown folder: {folder_name}")
        continue

    bucket_name = FOLDER_BUCKET_MAP[folder_name]
    ensure_bucket(bucket_name)
    # Upload files from this folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if os.path.isdir(file_path):
            continue

        upload_file(bucket_name, file_path, filename)

print("\nAll files uploaded successfully!")
