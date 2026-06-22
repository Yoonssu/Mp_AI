from pathlib import Path
import sys

from chromadb import PersistentClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings


def main():
    settings = get_settings()
    client = PersistentClient(path=str(settings.chroma_db_path))
    collection = client.get_collection(settings.detail_collection)

    print("--- 데이터베이스 요약 ---")
    print(f"상세 컬렉션: {settings.detail_collection}")
    print(f"현재 총 저장된 청크 개수: {collection.count()}")

    sample = collection.peek(1)
    if not sample.get("documents"):
        print("샘플 데이터가 없습니다.")
        return

    print("\n--- 데이터 샘플 확인 ---")
    print("메타데이터:")
    print(sample["metadatas"][0])
    print("\n텍스트 내용:")
    print(sample["documents"][0][:200] + "...")


if __name__ == "__main__":
    main()
