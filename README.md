# 금융 RAG 에이전트

은행 상품 약관 PDF를 ChromaDB에 저장하고, Streamlit UI에서 상품 추천, 우대 조건 검색, 만기 예상액 계산을 수행하는 RAG 프로젝트입니다.

## 구조

- `app/config.py`: 프로젝트 경로, 모델명, 컬렉션명 설정
- `app/agent/graph.py`: LangGraph 에이전트 워크플로우
- `app/agent/tools.py`: Chroma 검색 도구와 만기 금액 계산 도구
- `app_ui.py`: Streamlit 채팅 UI
- `workers/tasks.py`: PDF 약관 인제스션
- `workers/verify.py`: ChromaDB 데이터 확인
- `scripts/check_project.py`: 실행 전 프로젝트 상태 점검

## 설치

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

`.env`에 `UPSTAGE_API_KEY` 값을 입력하세요.

## 실행 전 점검

```powershell
python scripts/check_project.py
python workers/verify.py
```

## UI 실행

```powershell
streamlit run app_ui.py
```

## API 헬스 체크 실행

```powershell
uvicorn app.main:app --reload
```

브라우저에서 `http://127.0.0.1:8000/health`를 확인하면 됩니다.

## 문서 재인제스션

```powershell
python workers/tasks.py
```

이미 처리된 파일은 `product_summaries` 컬렉션의 `filename` 메타데이터 기준으로 건너뜁니다.
