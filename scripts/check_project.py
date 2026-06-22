from pathlib import Path
import sys

from chromadb import PersistentClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings  # noqa: E402
from app.agent.tools import calculate_maturity_amount  # noqa: E402


def check_path(label: str, path: Path) -> bool:
    ok = path.exists()
    print(f"[{'OK' if ok else 'FAIL'}] {label}: {path}")
    return ok


def main() -> int:
    settings = get_settings()
    checks = [
        check_path("project root", PROJECT_ROOT),
        check_path("storage dir", settings.storage_dir),
        check_path("chroma db dir", settings.chroma_db_path),
    ]

    pdf_count = len(list(settings.storage_dir.glob("*.pdf"))) if settings.storage_dir.exists() else 0
    print(f"[INFO] PDF files: {pdf_count}")
    checks.append(pdf_count > 0)

    calc_result = calculate_maturity_amount("300000, 6.0")
    print(f"[INFO] calculation smoke test: {calc_result}")
    checks.append("계산 성공" in calc_result)

    try:
        client = PersistentClient(path=str(settings.chroma_db_path))
        collections = {
            collection.name if hasattr(collection, "name") else str(collection)
            for collection in client.list_collections()
        }
        print(f"[INFO] Chroma collections: {sorted(collections)}")
        checks.append(settings.detail_collection in collections)
        checks.append(settings.summary_collection in collections)
    except Exception as exc:
        print(f"[FAIL] Chroma check failed: {exc}")
        checks.append(False)

    if all(checks):
        print("[OK] Project check passed.")
        return 0

    print("[FAIL] Project check failed. Review messages above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
