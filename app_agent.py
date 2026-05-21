import os
import time
import re
from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver 
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_chroma import Chroma
from langchain_upstage import UpstageEmbeddings, ChatUpstage

load_dotenv()

# =========================================================================
# 1. DB 연결 (Track A: 요약 컬렉션, Track B: 상세 컬렉션 이원화)
# =========================================================================
DB_PATH = "c:/Users/user/Documents/과천시/Mon/Mp_AI/chroma_db"
embeddings = UpstageEmbeddings(model="solar-embedding-1-large")

summary_vector_store = Chroma(collection_name="product_summaries", embedding_function=embeddings, persist_directory=DB_PATH)
detail_vector_store = Chroma(collection_name="financial_docs", embedding_function=embeddings, persist_directory=DB_PATH)


# =========================================================================
# 2. 하이브리드 RAG 전용 도구(Tools) 정의 (3번 교정 포인트: 인자 파싱 안정화)
# =========================================================================

def search_specific_details(input_str: str) -> str:
    """전수조사 후 후보로 뽑힌 특정 파일의 우대금리 세부 조건이나 자격 요건을 정밀 조회합니다."""
    try:
        # LLM이 콤마 대신 공백이나 다른 문자로 인자를 던질 경우를 대비한 방어 코드
        if ',' in input_str:
            filename, keyword = [x.strip() for x in input_str.split(',', 1)]
        else:
            filename = input_str.strip()
            keyword = "우대조건"
            
        print(f"\n⚡ [하이브리드 RAG] Phase 2: {filename} 파일 상세 내용 딥다이브 (키워드: {keyword})")
        
        retriever = detail_vector_store.as_retriever(
            search_kwargs={
                'k': 4,
                'filter': {'filename': filename}
            }
        )
        docs = retriever.invoke(keyword)
        
        if not docs:
            return f"[{filename}] 해당 키워드에 대한 상세 조항을 찾지 못했습니다."
            
        return "\n\n".join([f"[상세 조항 조각]\n{d.page_content}" for d in docs])
    except Exception as e:
        return f"상세 조회 실패. (에러: {str(e)})"


def calculate_maturity_amount(input_str: str) -> str:
    """월 납입액과 이자율을 받아 1년(12개월) 만기 단리 방식으로 세전 이자와 원금을 계산합니다."""
    try:
        # 이진 분류 및 숫자 추출 정규식 강화 (할루시네이션 숫자 전면 차단)
        clean_str = input_str.replace(" ", "").replace("%", "").replace("원", "")
        match = re.findall(r"[\d.]+", clean_str)
        
        if len(match) >= 2:
            amount = float(match[0])
            rate = float(match[1])
        else:
            amount, rate = map(float, clean_str.split(','))
            
        total_principal = amount * 12
        total_interest = sum(amount * (rate / 100) * (i / 12) for i in range(1, 13))
        
        return f"[계산 성공] 1년 만기 원금: {int(total_principal):,}원, 예상 세전 이자: {int(total_interest):,}원, 만기 총 수령액: {int(total_principal + total_interest):,}원"
    except Exception as e:
        return f"계산 실패. 정확한 숫자를 파싱할 수 없습니다. (에러: {str(e)})"


tools_map = {
    "search_specific_details": search_specific_details,
    "calculate_maturity_amount": calculate_maturity_amount
}

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# =========================================================================
# 3. Upstage 모델 설정 및 도구 규격 바인딩
# =========================================================================
llm = ChatUpstage(model="solar-1-mini-chat", temperature=0)
llm_with_tools = llm.bind_tools([
    {
        "type": "function",
        "function": {
            "name": "search_specific_details",
            "description": "선택한 특정 약관 파일의 우대금리 세부 조항을 정밀 파싱합니다. 인자는 반드시 '파일명,키워드' 구조여야 합니다. 예시: 'SH_Youth_01_Product_Manual.pdf,우대조건'",
            "parameters": {"type": "object", "properties": {"input_str": {"type": "string"}}, "required": ["input_str"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_maturity_amount",
            "description": "월 납입액과 금리로 만기 이자를 계산합니다. 인자는 반드시 '월납입액숫자,금리숫자' 형태로 넣어야 합니다. 예시: '300000,6.05'",
            "parameters": {"type": "object", "properties": {"input_str": {"type": "string"}}, "required": ["input_str"]}
        }
    }
])


# =========================================================================
# 4. LangGraph 그래프 노드 및 컨트롤 라우터 함수 정의
#    (1번 & 2번 교정 포인트: 파싱 버그 컷 및 연속 질문 맥락 토스 복원)
# =========================================================================

def force_first_inspection(state: AgentState):
    """최초 질문일 때만 1차 요약 RAG를 돌리고, 연속 질문 시에는 메시지를 오염시키지 않고 그대로 패스합니다."""
    
    # 💥 [2번 버그 해결]: 대화 기록이 1개보다 많다 = 연속 질문이다!
    # 기존 메시지들을 한 글자도 건드리지 않고 그대로 반환하여 agent 노드가 새 질문을 인식하게 합니다.
    if len(state["messages"]) > 1:
        print("ℹ️ [메모리 작동] 연속 질문 맥락이 감지되어 요약본 DB 조회를 안전하게 패스합니다.")
        return {"messages": []}

    print(f"\n⚡ [진짜 RAG] 사용자 질문 기반 벡터 의미 검색 가동 중...")
    
    # 💥 [1번 버그 해결]: 무리한 텍스트 자르기(split) 대신 안전하게 최신 질문 content 확보
    last_msg = state["messages"][-1]
    user_msg_content = last_msg.content.strip()
    
    try:
        print(f"▶️ [STEP 1: 임베딩] '{user_msg_content}' -> 벡터 변환 중...")
        query_vector = embeddings.embed_query(user_msg_content)
        print("▶️ [STEP 2: DB 검색] ChromaDB 유사도 대조 중...")
        docs = summary_vector_store.similarity_search_by_vector(query_vector, k=5) 
        print(f"✅ [STEP 2 완료] 가장 유사한 {len(docs)}개 문서 검색 성공!")
    except Exception as e:
        print(f"❌ [에러 발생] {str(e)}")
        raise e

    results = [f"📄 [{d.metadata.get('filename', 'Unknown')}] {d.page_content}" for d in docs]
    summary_context = "\n".join(results)
    
    # 첫 질문일 때만 컨텍스트를 이쁘게 보강하여 넘겨줍니다.
    enriched_content = (
        f"[참고 데이터 - 상위 5개 후보 상품 요약본]\n{summary_context}\n"
        f" 사용자 원본 질문: {user_msg_content}"
    )
    
    return {"messages": [HumanMessage(content=enriched_content, id=last_msg.id)]}


def call_model(state: AgentState):
    system_prompt = (
        "당신은 오직 제공된 [참고 데이터], [과거 대화 기록], 그리고 도구의 실행 결과 안에서만 정답을 찾는 전문 금융 자산 컨설턴트입니다.\n"
        "★절대 경고: 당신의 사전 학습 지식으로 이자를 계산하거나 금액을 지어내지 마세요. 반드시 'calculate_maturity_amount' 도구를 호출해 그 결과를 보고 쓰세요.\n\n"
        
        "★[⚠️ 필독 - 출력 포맷 가이드]\n"
        "도구 결과가 모두 수집된 최종 단계에서는 오직 아래의 3가지 섹션만 순서대로 출력해야 하며, 내용 도돌이표 중복이나 이 외의 사족은 절대 금지합니다.\n\n"
        "[분석 결과]\n"
        "(마크다운 표 형태로 한글 은행명, 상품명, 최고 금리, 가입 대상, 월 납입 한도, 툴이 계산해준 진짜 '만기 수령액' 정보 정리)\n\n"
        "[추천 이유]\n"
        "(해당 상품의 구체적인 우대 자격 조건들을 데이터에 기반하여 구체적으로 서술)\n\n"
        "[최종 분석 답변]\n"
        "(도구 결과로 나온 원금과 이자를 바탕으로 깔끔한 리포트 문장 마무리)"
    )
    
    clean_messages = []
    for m in state["messages"]:
        if isinstance(m, (HumanMessage, AIMessage, ToolMessage)):
            clean_messages.append(m)
            
    messages = [{"role": "system", "content": system_prompt}] + clean_messages
    response = llm_with_tools.invoke(messages)
    print("✅ [API 완료] Upstage LLM 응답 도착!")
    
    return {"messages": [response]}


def router(state: AgentState):
    last_message = state["messages"][-1]
    if len(state["messages"]) > 15:
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

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)