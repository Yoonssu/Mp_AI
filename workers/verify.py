from chromadb import PersistentClient

# 1. DB 연결 (저장된 경로와 동일해야 합니다)
client = PersistentClient(path="./chroma_db")
collection = client.get_collection("financial_docs")

# 2. 총 저장된 데이터 개수 확인
print(f"--- 데이터베이스 요약 ---")
print(f"현재 총 저장된 청크(Chunk) 개수: {collection.count()}")

# 3. 샘플 데이터(데이터 중 1개)를 꺼내서 확인
print(f"\n--- 데이터 샘플 확인 ---")
sample = collection.peek(1) # 가장 최신 혹은 첫 번째 데이터를 살짝 엿봅니다.

print("메타데이터(Metadata):")
print(sample['metadatas'][0])

print("\n텍스트 내용(Text):")
print(sample['documents'][0][:200] + "...") # 너무 길면 앞부분만 출력