import time
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from chromadb import PersistentClient
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_upstage import ChatUpstage, UpstageDocumentParseLoader, UpstageEmbeddings
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings


settings = get_settings()
client = PersistentClient(path=str(settings.chroma_db_path))

embeddings = UpstageEmbeddings(model=settings.embedding_model)
llm = ChatUpstage(model=settings.chat_model)

detail_vector_store = Chroma(
    collection_name=settings.detail_collection,
    embedding_function=embeddings,
    persist_directory=str(settings.chroma_db_path),
)
summary_vector_store = Chroma(
    collection_name=settings.summary_collection,
    embedding_function=embeddings,
    persist_directory=str(settings.chroma_db_path),
)


def get_bank_name(code: str) -> str:
    mapping = {"SH": "Shinhan", "KB": "Kookmin", "HN": "Hana", "TS": "Toss", "KA": "Kakao"}
    return mapping.get(code, "Unknown")


def parse_file_metadata(filename: str) -> dict[str, str]:
    parts = filename.replace(".pdf", "").split("_")
    if len(parts) < 2:
        raise ValueError(f"지원하지 않는 파일명 형식입니다: {filename}")

    return {
        "product_code": f"{parts[0]}_{parts[1]}",
        "bank": get_bank_name(parts[0]),
        "doc_type": parts[3] if len(parts) > 3 else "Unknown",
        "filename": filename,
    }


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def process_single_file(filename, markdown_splitter, text_splitter):
    metadata = parse_file_metadata(filename)
    print(f"[{metadata['bank']} - {metadata['product_code']}] 처리 중: {filename}")

    file_path = settings.storage_dir / filename
    loader = UpstageDocumentParseLoader(str(file_path), output_format="markdown")
    docs = loader.load()
    full_text = docs[0].page_content

    summary_prompt = PromptTemplate.from_template(
        "다음은 은행 상품 약관입니다. 이 상품의 '상품명', '최고 금리', "
        "'가입 대상', '월 납입 한도'를 2~3줄로 명확하게 요약하세요.\n\n약관 일부:\n{text}"
    )
    summary_result = (summary_prompt | llm).invoke({"text": full_text[:3000]})

    summary_vector_store.add_texts(texts=[summary_result.content], metadatas=[metadata])
    print("   요약본 저장 완료")

    md_header_chunks = markdown_splitter.split_text(full_text)
    final_chunks = text_splitter.split_documents(md_header_chunks)

    for chunk in final_chunks:
        chunk.metadata.update(metadata)

    detail_vector_store.add_documents(final_chunks)
    print(f"   상세 청크 {len(final_chunks)}개 저장 완료: {filename}")
    time.sleep(2)


def get_processed_files() -> set[str]:
    try:
        summary_collection = client.get_collection(settings.summary_collection)
        existing_docs = summary_collection.get()
    except Exception:
        return set()

    return {
        metadata["filename"]
        for metadata in existing_docs.get("metadatas") or []
        if metadata and "filename" in metadata
    }


def ingest_files():
    processed_files = get_processed_files()
    print(f"이미 처리된 파일: {len(processed_files)}개")

    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "h1"), ("##", "h2")])
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    for file_path in sorted(settings.storage_dir.glob("*.pdf")):
        if file_path.name in processed_files:
            print(f"건너뜀: {file_path.name}")
            continue
        process_single_file(file_path.name, markdown_splitter, text_splitter)


if __name__ == "__main__":
    print("하이브리드 RAG DB 인제스션 시작")
    ingest_files()
