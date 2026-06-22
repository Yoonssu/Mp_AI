from chromadb import PersistentClient

from app.config import get_settings


def get_chroma_client() -> PersistentClient:
    settings = get_settings()
    return PersistentClient(path=str(settings.chroma_db_path))
