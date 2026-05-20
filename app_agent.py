import os
import time
import re
from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_chroma import Chroma
from langchain_upstage import UpstageEmbeddings, ChatUpstage

load_dotenv()

# =========================================================================
# 1. DB 연결 (Track A: 요약 컬렉션, Track B: 상세 컬렉션 이원화)
# =========================================================================
DB_PATH = "c:/Users/user/Documents/과천시/Mon/Mp_AI/chroma_db"
embeddings = UpstageEmbeddings(model="solar-embedding-1-large")

# [Track A] 66개 파일의 핵심 요약문만 저장된 컬렉션 (총 청크 개수 딱 66개 내외)
summary_vector_store = Chroma(collection_name="product_summaries", embedding_function=embeddings, persist_directory=DB_PATH)
# [Track B] 기존 약관 세부 조항이 잘게 쪼개져 있는 컬렉션 (수천 개 청크)
detail_vector_store = Chroma(collection_name="financial_docs", embedding_function=embeddings, persist_directory=DB_PATH)


# =========================================================================
# 2. 하이브리드 RAG 전용 도구(Tools) 정의
# =========================================================================

def search_specific_details(input_str: str) -> str:
    """전수조사 후 후보로 뽑힌 특정 파일의 우대금리 세부 조건이나 자격 요건을 정밀 조회합니다.
    입력 형식은 반드시 '파일명, 검색키워드'여야 합니다. (예: 'HN_RR_01_Product_Manual.pdf, 우대금리')"""
    try:
        filename, keyword = [x.strip() for x in input_str.split(',')]
        print(f"\n⚡ [하이브리드 RAG] Phase 2: {filename} 파일 상세 내용 딥다이브 (키워드: {keyword})")
        
        retriever = detail_vector_store.as_retriever(
            search_kwargs={
                'k': 4,
                'filter': {'filename': filename}
            }
        )
        docs = retriever.invoke(keyword)
        
        results = []
        for d in docs:
            results.append(f"[상세 조항 조각]\n{d.page_content}")
        return "\n\n".join(results)
    except Exception as e:
        return f"상세 조회 실패. 인자 형식을 '파일명,키워드' 형태로 정확히 넘겨야 합니다. (에러: {str(e)})"


def calculate_maturity_amount(input_str: str) -> str:
    """월 납입액과 이자율을 받아 1년(12개월) 만기 단리 방식으로 세전 이자와 원금을 계산합니다.
    입력 형식은 반드시 '월납입액,금리' 숫자로만 이루어져야 합니다. (예: '100000,4.5')"""
    try:
        clean_str = input_str.replace(" ", "").replace("%", "")
        match = re.search(r"(\d+),([\d.]+)", clean_str)
        if match:
            amount = float(match.group(1))
            rate = float(match.group(2))
        else:
            amount, rate = map(float, clean_str.split(','))
            
        total_principal = amount * 12
        total_interest = sum(amount * (rate / 100) * (i / 12) for i in range(1, 13))
        
        return f"[계산 성공] 1년 만기 원금: {int(total_principal):,}원, 예상 세전 이자: {int(total_interest):,}원, 만기 총 수령액: {int(total_principal + total_interest):,}원"
    except Exception as e:
        return f"계산 실패. 에이전트 너는 절대로 수식이나 문장을 쓰지 말고 오직 '100000,4.5' 형태로 숫자만 인자로 다시 넘겨라. (에러: {str(e)})"


tools_map = {
    "search_specific_details": search_specific_details,
    "calculate_maturity_amount": calculate_maturity_amount
}

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# =========================================================================
# 3. Upstage 모델 설정 및 도구 규격 바인딩 (엄격화)
# =========================================================================
llm = ChatUpstage(model="solar-1-mini-chat", temperature=0)
llm_with_tools = llm.bind_tools([
    {
        "type": "function",
        "function": {
            "name": "search_specific_details",
            "description": "선택한 특정 약관 파일의 우대금리 세부 조항이나 조건 서류를 정밀 파싱합니다. 인자는 반드시 '파일명,키워드' 구조로 콤마로 구분된 단일 문자열이어야 합니다. (예: 'HN_MS_01_Product_Manual.pdf,우대이율')",
            "parameters": {"type": "object", "properties": {"input_str": {"type": "string"}}, "required": ["input_str"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_maturity_amount",
            "description": "★절대 경고: 인자에 %, *, 연산 기호나 글자를 포함하지 마십시오. 오직 '월납입액숫자,금리숫자'만 순수하게 넣어야 합니다. 예시: 10만 원에 5.2% 이자 계산을 원하면 무조건 '100000,5.2' 라고만 적어 호출하십시오.",
            "parameters": {"type": "object", "properties": {"input_str": {"type": "string"}}, "required": ["input_str"]}
        }
    }
])


# =========================================================================
# 4. LangGraph 그래프 노드 및 컨트롤 라우터 함수 정의
# =========================================================================

def force_first_inspection(state: AgentState):
    """[디버깅 모드] 임베딩 API와 DB 검색 구간을 쪼개어 무한 대기 구간을 찾습니다."""
    print(f"\n⚡ [진짜 RAG] 사용자 질문 기반 벡터 의미 검색 가동 중...")
    
    # 🔍 1. API 키 로드 검사 (무한 재시도 버그 방지)
    api_key = os.getenv("UPSTAGE_API_KEY")
    if not api_key:
        print("❌ [경고/에러] UPSTAGE_API_KEY가 없습니다! .env 파일이 제대로 로드되지 않아 API가 무한 대기 중일 수 있습니다.")
    else:
        print(f"✅ API 키 인식 완료 (시작 문자: {api_key[:5]}...)")

    last_msg = state["messages"][-1]
    user_msg_content = last_msg.content.replace("[사용자 원본 질문]:", "").split("===")[0].strip()
    
    try:
        # 🔍 2. Upstage 임베딩 API 통신 테스트 (대부분 여기서 멈춤)
        print(f"▶️ [STEP 1: 임베딩] '{user_msg_content}' -> 수치(벡터)로 변환 시도 중...")
        query_vector = embeddings.embed_query(user_msg_content)
        print("✅ [STEP 1 완료] Upstage 서버에서 수치 변환 성공!")
        
        # 🔍 3. Chroma DB 로컬 검색 테스트
        print("▶️ [STEP 2: DB 검색] 변환된 수치로 ChromaDB 유사도 대조 중...")
        # invoke 대신 쪼개진 함수 직접 사용
        docs = summary_vector_store.similarity_search_by_vector(query_vector, k=5) 
        print(f"✅ [STEP 2 완료] 가장 유사한 {len(docs)}개 문서 검색 성공!")
        
    except Exception as e:
        print(f"❌ [에러 발생] {str(e)}")
        raise e

    results = [f"📄 [{d.metadata.get('filename', 'Unknown')}] {d.page_content}" for d in docs]
    summary_context = "\n".join(results)
    
    enriched_content = (
        f"[사용자 원본 질문]: {user_msg_content}\n\n"
        f"=== [의미 기반 벡터 검색 결과: 상위 5개 후보 상품 요약본] ===\n"
        f"{summary_context}\n"
        f"====================================="
    )
    
    enriched_message = HumanMessage(content=enriched_content, id=last_msg.id)
    return {"messages": [enriched_message]}


def call_model(state: AgentState):
    system_prompt = (
        "당신은 오직 제공된 [참고 데이터] 안에서만 정답을 찾는 깐깐한 금융 비서입니다.\n"
        "★절대 경고: 당신의 사전 학습 지식을 사용하거나 '하나원큐' 같은 가상의 상품을 지어내지 마세요.\n"
        "무조건 사용자 메시지 아래에 주입된 [참고 데이터: 4대 은행 전체 상품 요약본] 텍스트 안에 존재하는 '실제 상품명'과 '실제 파일명'만 골라서 사용해야 합니다.\n\n"
        
        "안내 문구를 출력하지 말고, 찾은 실제 파일명을 사용해 즉시 다음 도구를 순서대로 호출하세요:\n"
        "1. search_specific_details (인자 예시: '실제파일명.pdf, 우대조건')\n"
        "2. calculate_maturity_amount"
    )
    # 400 에러 재발을 완벽하게 예방하기 위해 시스템 프롬프트 결합 시 순수한 기록들만 필터링
    clean_messages = []
    for m in state["messages"]:
        if isinstance(m, HumanMessage):
            # HumanMessage는 tool_calls 검사 없이 안전하게 추가
            clean_messages.append(m)
        elif isinstance(m, AIMessage):
            # AIMessage 일 때만 내부 tool_calls 속성이 있는지 검사
            clean_messages.append(m)
        elif isinstance(m, ToolMessage):
            clean_messages.append(m)
            
    messages = [{"role": "system", "content": system_prompt}] + clean_messages
    response = llm_with_tools.invoke(messages)
    print("✅ [API 완료] Upstage LLM 응답 도착!")
    # -----------------------------
    
    return {"messages": [response]}

def router(state: AgentState):
    last_message = state["messages"][-1]
    if len(state["messages"]) > 12:
        return "end"
    if last_message.tool_calls:
        return "continue"
    return "end"


def call_tools(state: AgentState):
    last_message = state["messages"][-1]
    tool_messages = []
    
    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        arg_value = tool_args.get("input_str") or tool_args.get("query")
        
        print(f"\n[🔄 Action] 에이전트 도구 호출 승인: {tool_name} (전달된 인자: '{arg_value}')")
        
        result = tools_map[tool_name](arg_value)
        tool_messages.append(ToolMessage(content=result, tool_call_id=tool_call["id"], name=tool_name))
        
    return {"messages": tool_messages}


# =========================================================================
# 5. 워크플로우 그래프 빌드 및 컴파일
# =========================================================================
workflow = StateGraph(AgentState)

workflow.add_node("force_inspect", force_first_inspection)
workflow.add_node("agent", call_model)
workflow.add_node("tools", call_tools)

workflow.add_edge(START, "force_inspect")
workflow.add_edge("force_inspect", "agent") 

workflow.add_conditional_edges("agent", router, {"continue": "tools", "end": END})
workflow.add_edge("tools", "agent")

app = workflow.compile()

if __name__ == "__main__":
    print("\n==================================================")
    print("🚀 하이브리드 2-Track RAG 금융 에이전트 시스템 가동")
    print("==================================================\n")
    
    while True:
        user_query = input("[User 질문 입력]: ")
        if user_query.strip() in ["종료", "exit", "quit"]:
            break
        if not user_query.strip():
            continue
            
        inputs = {"messages": [HumanMessage(content=user_query)]}
        for output in app.stream(inputs, stream_mode="updates"):
            for node_name, state_update in output.items():
                if node_name == "agent":
                    last_msg = state_update["messages"][-1]
                    if last_msg.content:
                        print(f"\n[✨ 에이전트 최종 분석 답변]\n{last_msg.content}\n")
        print("-" * 60)