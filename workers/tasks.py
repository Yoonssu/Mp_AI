import os
import time
from chromadb import PersistentClient
from langchain_chroma import Chroma
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_upstage import UpstageEmbeddings, UpstageDocumentParseLoader, ChatUpstage
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

# 1. 설정 
DB_PATH = "c:/Users/user/Documents/과천시/Mon/Mp_AI/chroma_db"
STORAGE_DIR = "app/storage"

client = PersistentClient(path=DB_PATH)

embeddings = UpstageEmbeddings(model="solar-embedding-1-large")
llm = ChatUpstage(model="solar-1-mini-chat")

# Track B: 상세 조항 파편들 저장용
detail_vector_store = Chroma(collection_name="financial_docs", embedding_function=embeddings, persist_directory=DB_PATH)
# Track A: 1단계 전수조사용 요약본 저장용
summary_vector_store = Chroma(collection_name="product_summaries", embedding_function=embeddings, persist_directory=DB_PATH)

def get_bank_name(code):
    mapping = {"SH": "Shinhan", "KB": "Kookmin", "HN": "Hana", "TS": "Toss", "KA": "Kakao"}
    return mapping.get(code, "Unknown")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def process_single_file(filename, markdown_splitter, text_splitter):
    parts = filename.replace(".pdf", "").split("_")
    product_code = f"{parts[0]}_{parts[1]}"
    bank_name = get_bank_name(parts[0])
    doc_type = parts[3] if len(parts) > 3 else "Unknown"

    print(f"[{bank_name} - {product_code}] 처리 중: {filename}")

    file_path = os.path.join(STORAGE_DIR, filename)
    loader = UpstageDocumentParseLoader(file_path, output_format="markdown")
    docs = loader.load()
    full_text = docs[0].page_content

    # =========================================================================
    # LLM 요약본(Summary) 생성 및 Track A 저장
    # =========================================================================
    print(f"   -> LLM 요약본(Summary) 생성 중...")
    summary_prompt = PromptTemplate.from_template(
        "다음은 은행 상품 약관입니다. 이 상품의 '상품명', '최고 금리', '가입 대상(예: 사회초년생 등)', '월 납입 한도'를 2~3줄로 명확하게 요약하세요.\n\n약관일부:\n{text}"
    )
    summary_chain = summary_prompt | llm
    summary_result = summary_chain.invoke({"text": full_text[:3000]})
    
    summary_vector_store.add_texts(
        texts=[summary_result.content],
        metadatas=[{"product_code": product_code, "bank": bank_name, "filename": filename}]
    )
    print(f"   ✅ 요약본 저장 완료")

    # =========================================================================
    # 기존 로직: 1000글자씩 쪼개서 상세 컬렉션(Track B)에 저장
    # =========================================================================
    md_header_chunks = markdown_splitter.split_text(full_text)
    final_chunks = text_splitter.split_documents(md_header_chunks)
    
    for chunk in final_chunks:
        chunk.metadata.update({
            "product_code": product_code,
            "bank": bank_name,
            "doc_type": doc_type,
            "filename": filename
        })
    
    detail_vector_store.add_documents(final_chunks)
    print(f"   ✅ 상세 파편 {len(final_chunks)}개 저장 완료: {filename}")
    time.sleep(2)

def ingest_files():
    # [핵심 중복 방지 로직]
    # 이미 성공적으로 저장된 파일 이름 리스트 가져오기 (Track A 기준)
    processed_files = set()
    try:
        # ChromaDB 클라이언트를 통해 직접 product_summaries 컬렉션 로드
        summary_collection = client.get_collection("product_summaries")
        existing_docs = summary_collection.get()
        if existing_docs['metadatas']:
            processed_files = set(m['filename'] for m in existing_docs['metadatas'] if 'filename' in m)
            print(f"ℹ️ 기존 DB에서 이미 처리 완료된 파일 {len(processed_files)}개를 감지했습니다.")
    except Exception:
        # 컬렉션이 아예 없거나 비어있는 최초 실행 시점 처리
        print("ℹ️ 기존에 처리된 내역이 없거나 새 컬렉션입니다. 처음부터 시작합니다.")
    
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "h1"), ("##", "h2")])
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    for filename in os.listdir(STORAGE_DIR):
        if not filename.endswith(".pdf"): continue
        
        # [핵심] 이미 처리된 파일 건너뛰기
        if filename in processed_files:
            print(f"⏭️ 건너뜀 (이미 저장됨): {filename}")
            continue
            
        process_single_file(filename, markdown_splitter, text_splitter)

if __name__ == "__main__":
    print("하이브리드 RAG DB '이어받기' 인제스션 가동!")
    ingest_files()