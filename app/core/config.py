# app/core/config.py

import os
from typing import ClassVar, Optional
from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv
load_dotenv()

class Settings(BaseSettings):
    """
    Main application settings (Pydantic v2 compatible).
    Only real config fields are typed normally.
    Pure constants are marked as ClassVar.
    """

    # --------------------------------------------------------------------
    # Core app settings
    # --------------------------------------------------------------------
    ENV: str = Field(default="dev")
    DEBUG: bool = Field(default=False)
    APP_NAME: str = Field(default="financial-rag-demo")

    # --------------------------------------------------------------------
    # Database
    # --------------------------------------------------------------------
    POSTGRES_HOST: str = Field(os.getenv("POSTGRES_HOST", "localhost"))
    POSTGRES_PORT: int = Field(int(os.getenv("POSTGRES_PORT", 5432)))
    POSTGRES_DB: str = Field(os.getenv("POSTGRES_DB", "stocks"))
    POSTGRES_USER: str = Field(os.getenv("POSTGRES_USER", "postgres"))
    POSTGRES_PASSWORD: str = Field(os.getenv("POSTGRES_PASSWORD", "postgres"))
    # --------------------------------------------------------------------
    # MinIO / object storage
#     --------------------------------------------------------------------
    MINIO_ACCESS_KEY: str = os.getenv("MINIO_ACCESS_KEY")
    MINIO_SECRET_KEY: str = os.getenv("MINIO_SECRET_KEY")
    MINIO_ENDPOINT: str = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    MINIO_BUCKET_DOCS: str = "docs"
    MINIO_BUCKET_PPT_IMAGES: str = "ppt-images"

    # --------------------------------------------------------------------
    # Langfuse (keys from env), base URL as constant
    # --------------------------------------------------------------------
    LANGFUSE_SECRET_KEY: Optional[str] = os.getenv("LANGFUSE_SECRET_KEY")
    LANGFUSE_PUBLIC_KEY: Optional[str] = os.getenv("LANGFUSE_PUBLIC_KEY")

    # CONSTANT (ignored by Pydantic as field)
    LANGFUSE_HOST: ClassVar[str] = os.getenv("LANGFUSE_HOST", "https://us.cloud.langfuse.com")

    # --------------------------------------------------------------------
    # Paths (include DATA_FOLDER to match old env)
    # --------------------------------------------------------------------
    DATA_FOLDER: str = Field(os.getenv("DATA_FOLDER"))
    CHROMA_PERSIST_DIR: str = Field(os.getenv("CHROMA_PERSIST_DIR"))

    # Optionally: QA dataset dir if you still use it
    QA_DATASET_DIR: str = Field(os.getenv("QA_DATASET_DIR"))
    
    EVALUATION_DIR: str = Field(os.getenv("EVALUATION_DIR"))
    # --------------------------------------------------------------------
    # Evaluation defaults (if you want them here; otherwise remove)
    # --------------------------------------------------------------------

    QA_DATA_PATH: str = Field(os.getenv("QA_DATA_PATH"))

    EVALUATION_SAMPLE_SIZE: int = 0  # 0 = all

    INDEX_BUILDER_GENERATE_QA: bool = True
    INDEX_BUILDER_QA_MIN_CHUNK_TOKENS: int =200
    INDEX_BUILDER_CHUNK_MAX_TOKENS: int =512
    INDEX_BUILDER_CHUNK_OVERLAP_TOKENS: int =50
    INDEX_BUILDER_USE_SENTENCE_CHUNKING: bool = True
    
    OUTPUT_DIR: str = Field(os.getenv("OUTPUT_DIR"))
    # --------------------------------------------------------------------
    # Gemini / LLM
    # --------------------------------------------------------------------
    GEMINI_MODEL: str = Field(os.getenv("GEMINI_MODEL"))
    GEMINI_API_KEYS: list[str] = Field(os.getenv("GEMINI_API_KEYS", "").split(","))
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "allow"  


settings = Settings()
