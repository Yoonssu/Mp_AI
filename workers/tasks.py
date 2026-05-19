import os
import time
from chromadb import PersistentClient
from langchain_chroma import Chroma
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_upstage import UpstageEmbeddings, UpstageDocumentParseLoader
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

# 1. 설정
DB_PATH = "./chroma_db"
STORAGE_DIR = "app/storage"

client = PersistentClient(path=DB_PATH)
embeddings = UpstageEmbeddings(model="solar-embedding-1-large")
vector_store = Chroma(collection_name="financial_docs", embedding_function=embeddings, persist_directory=DB_PATH)

def get_bank_name(code):
    mapping = {"SH": "Shinhan", "KB": "Kookmin", "HN": "Hana", "TS": "Toss", "KA": "Kakao"}
    return mapping.get(code, "Unknown")

# 재시도 로직이 적용된 개별 파일 처리 함수
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def process_single_file(filename, markdown_splitter, text_splitter):
    parts = filename.replace(".pdf", "").split("_")
    product_code = f"{parts[0]}_{parts[1]}"
    bank_name = get_bank_name(parts[0])
    doc_type = parts[3] if len(parts) > 3 else "Unknown"

    print(f"[{bank_name} - {product_code}] 처리 중: {filename}")

    # 1. 문서 로드
    file_path = os.path.join(STORAGE_DIR, filename)
    loader = UpstageDocumentParseLoader(file_path, output_format="markdown")
    docs = loader.load()

    # 2. 청킹
    md_header_chunks = markdown_splitter.split_text(docs[0].page_content)
    final_chunks = text_splitter.split_documents(md_header_chunks)
    
    # 3. 메타데이터 주입
    for chunk in final_chunks:
        chunk.metadata.update({
            "product_code": product_code,
            "bank": bank_name,
            "doc_type": doc_type,
            "filename": filename
        })
    
    # 4. DB 저장
    vector_store.add_documents(final_chunks)
    print(f"저장 완료: {filename}")
    time.sleep(2) # API 제한 준수

def ingest_files():
    # 저장된 파일 목록 확인
    existing_docs = collection.get()
    processed_files = set(existing_docs['metadatas'][i]['filename'] for i in range(len(existing_docs['metadatas'])))
    
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "h1"), ("##", "h2")])
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    for filename in os.listdir(STORAGE_DIR):
        if not filename.endswith(".pdf"): continue
        if filename in processed_files:
            print(f"건너뜀: {filename}")
            continue
        
        process_single_file(filename, markdown_splitter, text_splitter)

if __name__ == "__main__":
    ingest_files()