from fastapi import FastAPI

from app.config import get_settings


settings = get_settings()
api = FastAPI(title=settings.app_name)


@api.get("/health")
def health_check():
    return {
        "status": "ok",
        "chroma_db_path": str(settings.chroma_db_path),
        "storage_dir": str(settings.storage_dir),
    }


app = api
