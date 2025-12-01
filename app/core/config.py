# app/core/config.py

import os
from typing import ClassVar, Optional
from pydantic_settings import BaseSettings
from pydantic import Field


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
    POSTGRES_HOST: str = Field(default="localhost")
    POSTGRES_PORT: int = Field(default=5432)
    POSTGRES_DB: str = Field(default="stocks")
    POSTGRES_USER: str = Field(default="postgres")
    POSTGRES_PASSWORD: str = Field(default="postgres")

    # --------------------------------------------------------------------
    # MinIO / object storage
#     --------------------------------------------------------------------
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_BUCKET_DOCS: str = "docs"
    MINIO_BUCKET_PPT_IMAGES: str = "ppt-images"

    # --------------------------------------------------------------------
    # Langfuse (keys from env), base URL as constant
    # --------------------------------------------------------------------
    LANGFUSE_SECRET_KEY: Optional[str] = "sk-lf-5a8f144f-da4d-49a3-b084-fc7eb1aeb73d"
    LANGFUSE_PUBLIC_KEY: Optional[str] =  "pk-lf-065cb455-329f-4dbe-8c48-a197ad95a6df"

    # CONSTANT (ignored by Pydantic as field)
    LANGFUSE_HOST: ClassVar[str] = "https://us.cloud.langfuse.com"

    # --------------------------------------------------------------------
    # Paths (include DATA_FOLDER to match old env)
    # --------------------------------------------------------------------
    DATA_FOLDER: str = Field(
        default=r"D:\POC\Multimodal-Market-Intelligence-System\data"
    )
    CHROMA_PERSIST_DIR: str = Field(
        default=r"D:\POC\Multimodal-Market-Intelligence-System\chroma_data"
    )

    # Optionally: QA dataset dir if you still use it
    QA_DATASET_DIR: str = Field(
        default=r"C:\Users\Prajakta.Satav\Downloads\team2\demo\data\qa_datasets"
    )
    EVALUATION_DIR: str = Field(
        default=r"D:\POC\Multimodal-Market-Intelligence-System\evaluation"
    )
    # --------------------------------------------------------------------
    # Evaluation defaults (if you want them here; otherwise remove)
    # --------------------------------------------------------------------
    # QA_DATA_PATH: str = (
    #     r"C:\Users\Prajakta.Satav\Downloads\team2\demo\data\qa_datasets\golden_dataset.json"
    # )

    QA_DATA_PATH: str = (
        r"C:\Users\Prajakta.Satav\Downloads\team2\demo\data\qa_datasets\qa_dataset.json"
    )
    EVALUATION_SAMPLE_SIZE: int = 0  # 0 = all

    INDEX_BUILDER_GENERATE_QA: bool = True
    INDEX_BUILDER_QA_MIN_CHUNK_TOKENS: int =200
    INDEX_BUILDER_CHUNK_MAX_TOKENS: int =512
    INDEX_BUILDER_CHUNK_OVERLAP_TOKENS: int =50
    INDEX_BUILDER_USE_SENTENCE_CHUNKING: bool = True
    
    OUTPUT_DIR: str = Field(r"C:\Users\Prajakta.Satav\Downloads\team2\demo\evaluation_results")
    # --------------------------------------------------------------------
    # Gemini / LLM
    # --------------------------------------------------------------------
    #GEMINI_API_KEY: Optional[str] = "AIzaSyCmSry5QpOXV7r5DGy0IiZHvJ6Yyoto1yI"
    GEMINI_MODEL: str = Field(default="gemini-2.5-flash")
    GEMINI_API_KEYS: list[str] = ["AIzaSyAWMjLuQvByHXJHwnBnXEKFqA8k7vradHI", "AIzaSyCCBn5i3-jXyZdYC5gZ9WkYyhqssTaqjWs", "AIzaSyDH6-E4cHmQ1YCxXKaXO0wKLVbWs2KgaqM", "AIzaSyDJoiv-I6ED-gZfqE7XFkiBsHVnnTGi3Co", "AIzaSyAxaNswTM6D-a83uXlDSEM6_7bQ8rQaev4","AIzaSyBjZcThi5TV7PCBwQLN_Bz3hfY6Fwa-XPw", "AIzaSyBYG0hlhhiSGPFD1ZmEyKGkedUmdaqmoNg", "AIzaSyBpDUit0GZtmT9WhSaJBOZ4Z6t3M2IkUxs", "AIzaSyASUkc2O7vydX3S94OQZVD_gSLuaBcab48"]
 

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "allow"  # <-- IMPORTANT: ignore unknown env vars instead of failing


settings = Settings()
