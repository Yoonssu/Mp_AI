from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")


class Settings(BaseSettings):
    app_name: str = "Financial RAG Assistant"
    debug: bool = False

    chroma_db_path: Path = BASE_DIR / "chroma_db"
    storage_dir: Path = BASE_DIR / "app" / "storage"

    embedding_model: str = "solar-embedding-1-large"
    chat_model: str = "solar-1-mini-chat"
    summary_collection: str = "product_summaries"
    detail_collection: str = "financial_docs"

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        env_prefix="MP_AI_",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
