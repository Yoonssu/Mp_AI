import re

from langchain_chroma import Chroma
from langchain_upstage import UpstageEmbeddings

from app.config import get_settings


settings = get_settings()
embeddings = UpstageEmbeddings(model=settings.embedding_model)

summary_vector_store = Chroma(
    collection_name=settings.summary_collection,
    embedding_function=embeddings,
    persist_directory=str(settings.chroma_db_path),
)
detail_vector_store = Chroma(
    collection_name=settings.detail_collection,
    embedding_function=embeddings,
    persist_directory=str(settings.chroma_db_path),
)


def search_summary_candidates(query: str, limit: int = 5) -> str:
    query_vector = embeddings.embed_query(query)
    docs = summary_vector_store.similarity_search_by_vector(query_vector, k=limit)
    if not docs:
        return "검색된 상품 요약이 없습니다. 먼저 문서 인제스션을 실행해 주세요."
    return "\n".join(
        f"- [{doc.metadata.get('filename', 'Unknown')}] {doc.page_content}"
        for doc in docs
    )


def extract_numbers(input_str: str) -> list[float]:
    values = []
    for match in re.finditer(r"(\d[\d,]*(?:\.\d+)?)\s*(만원|만 원|천원|천 원|원|%)?", input_str):
        value = float(match.group(1).replace(",", ""))
        unit = match.group(2) or ""
        if "만" in unit:
            value *= 10000
        elif "천" in unit:
            value *= 1000
        values.append(value)
    return values


def extract_monthly_amount(input_str: str) -> float | None:
    monthly_patterns = [
        r"월\s*(\d[\d,]*(?:\.\d+)?)\s*(만원|만 원|천원|천 원|원)?",
        r"(\d[\d,]*(?:\.\d+)?)\s*(만원|만 원|천원|천 원|원)\s*씩",
    ]
    for pattern in monthly_patterns:
        match = re.search(pattern, input_str)
        if match:
            value = float(match.group(1).replace(",", ""))
            unit = match.group(2) or "원"
            if "만" in unit:
                value *= 10000
            elif "천" in unit:
                value *= 1000
            return value
    numbers = extract_numbers(input_str)
    return numbers[0] if numbers else None


def extract_annual_rate(input_str: str) -> float | None:
    rate_patterns = [
        r"최고\s*금리[^0-9]*(\d+(?:\.\d+)?)\s*%",
        r"연\s*(\d+(?:\.\d+)?)\s*%",
        r"(\d+(?:\.\d+)?)\s*%",
    ]
    for pattern in rate_patterns:
        match = re.search(pattern, input_str)
        if match:
            return float(match.group(1))
    return None


def search_specific_details(input_str: str) -> str:
    """Look up detailed clauses in a selected product document."""
    try:
        if "," in input_str:
            filename, keyword = [value.strip() for value in input_str.split(",", 1)]
        else:
            filename = input_str.strip()
            keyword = "우대조건"

        retriever = detail_vector_store.as_retriever(
            search_kwargs={"k": 4, "filter": {"filename": filename}}
        )
        docs = retriever.invoke(keyword)

        if not docs:
            return f"[{filename}] '{keyword}'에 대한 상세 조항을 찾지 못했습니다."

        return "\n\n".join(f"[상세 조항]\n{doc.page_content}" for doc in docs)
    except Exception as exc:
        return f"상세 조회 실패: {exc}"


def calculate_maturity_amount(input_str: str) -> str:
    """Calculate 12-month simple-interest maturity amount from monthly amount and annual rate."""
    try:
        numbers = extract_numbers(input_str)
        if len(numbers) < 2:
            return (
                "[계산 재시도 필요] 월 납입액과 연 금리가 모두 필요합니다. "
                "후보 상품의 최고 금리를 먼저 확인한 뒤 '월납입액, 연금리' 형식으로 다시 호출하세요. "
                "예: '200000, 5.0'"
            )

        monthly_amount = numbers[0]
        annual_rate = numbers[1]
        principal = monthly_amount * 12
        interest = sum(monthly_amount * (annual_rate / 100) * (month / 12) for month in range(1, 13))
        maturity_amount = principal + interest

        return (
            "[계산 성공] "
            f"1년 만기 원금: {int(principal):,}원, "
            f"예상 세전 이자: {int(interest):,}원, "
            f"만기 총 수령액: {int(maturity_amount):,}원"
        )
    except Exception as exc:
        return f"계산 실패. '300000, 6.05'처럼 월 납입액과 연 금리를 입력해 주세요. 오류: {exc}"


TOOLS_MAP = {
    "search_specific_details": search_specific_details,
    "calculate_maturity_amount": calculate_maturity_amount,
}

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "search_specific_details",
            "description": (
                "선택한 특정 약관 파일의 우대금리, 가입 대상, 납입 한도 같은 상세 조항을 조회합니다. "
                "인자는 '파일명,키워드' 형식입니다. 예: 'SH_Youth_01_Product_Manual.pdf,우대조건'"
            ),
            "parameters": {
                "type": "object",
                "properties": {"input_str": {"type": "string"}},
                "required": ["input_str"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_maturity_amount",
            "description": (
                "월 납입액과 연 금리로 12개월 단리 기준 만기 원금, 세전 이자, 총 수령액을 계산합니다. "
                "인자는 '월납입액,연금리' 형식입니다. 예: '300000,6.05'"
            ),
            "parameters": {
                "type": "object",
                "properties": {"input_str": {"type": "string"}},
                "required": ["input_str"],
            },
        },
    },
]
