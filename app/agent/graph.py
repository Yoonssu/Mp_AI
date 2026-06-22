from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_upstage import ChatUpstage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.config import get_settings
from app.agent.state import AgentState
from app.agent.tools import (
    TOOL_DEFINITIONS,
    TOOLS_MAP,
    calculate_maturity_amount,
    extract_annual_rate,
    extract_monthly_amount,
    search_summary_candidates,
)


settings = get_settings()
llm = ChatUpstage(model=settings.chat_model, temperature=0)
llm_with_tools = llm.bind_tools(TOOL_DEFINITIONS)


SYSTEM_PROMPT = """
당신은 제공된 참고 데이터, 과거 대화, 도구 실행 결과에 근거해 답하는 금융 상품 RAG 어시스턴트입니다.
근거가 부족하면 추측하지 말고 어떤 데이터가 부족한지 말하세요.

계산 규칙:
- 만기 수령액은 반드시 calculate_maturity_amount 도구 결과를 사용하세요.
- calculate_maturity_amount를 호출하기 전에는 월 납입액과 연 금리 두 값을 모두 확정해야 합니다.
- 사용자가 "월 20만원"처럼 납입액만 말하면, 참고 데이터의 후보 상품에서 최고 금리를 먼저 읽고 '200000, 금리' 형식으로 계산하세요.
- 도구 결과가 "[계산 재시도 필요]"로 시작하면 최종 답변을 하지 말고, 참고 데이터에서 금리를 확인해 calculate_maturity_amount를 다시 호출하세요.
- 금리가 참고 데이터에 없으면 계산하지 말고 어떤 정보가 부족한지 설명하세요.

최종 답변은 아래 섹션만 사용합니다.

[분석 결과]
상품명, 은행명, 최고 금리, 가입 대상, 월 납입 한도, 만기 수령액을 표로 정리합니다.

[추천 이유]
상세 조항과 우대 조건을 근거 중심으로 설명합니다.

[최종 분석 답변]
사용자가 바로 이해할 수 있게 결론을 간단히 정리합니다.
""".strip()


def enrich_first_turn(state: AgentState):
    if len(state["messages"]) > 1:
        return {"messages": []}

    last_message = state["messages"][-1]
    user_query = last_message.content.strip()
    summary_context = search_summary_candidates(user_query)
    enriched_content = (
        "[참고 데이터 - 상위 후보 상품 요약]\n"
        f"{summary_context}\n\n"
        f"[사용자 질문]\n{user_query}"
    )
    return {"messages": [HumanMessage(content=enriched_content, id=last_message.id)]}


def call_model(state: AgentState):
    clean_messages = [
        message
        for message in state["messages"]
        if isinstance(message, (HumanMessage, AIMessage, ToolMessage))
    ]
    response = llm_with_tools.invoke([{"role": "system", "content": SYSTEM_PROMPT}] + clean_messages)
    return {"messages": [response]}


def route_after_model(state: AgentState):
    last_message = state["messages"][-1]
    if len(state["messages"]) > 15:
        return "end"
    if getattr(last_message, "tool_calls", None):
        return "continue"
    if should_repair_calculation(state):
        return "repair"
    return "end"


def call_tools(state: AgentState):
    last_message = state["messages"][-1]
    tool_messages = []

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call.get("args", {})
        arg_value = tool_args.get("input_str") or tool_args.get("query") or ""
        tool = TOOLS_MAP.get(tool_name)
        result = tool(arg_value) if tool else f"등록되지 않은 도구입니다: {tool_name}"
        tool_messages.append(
            ToolMessage(content=result, tool_call_id=tool_call["id"], name=tool_name)
        )

    return {"messages": tool_messages}


def should_repair_calculation(state: AgentState) -> bool:
    last_message = state["messages"][-1]
    content = getattr(last_message, "content", "") or ""
    if "[자동 계산 보정]" in "\n".join(getattr(message, "content", "") for message in state["messages"]):
        return False
    return "[계산 재시도 필요]" in content


def repair_calculation(state: AgentState):
    last_content = getattr(state["messages"][-1], "content", "") or ""
    all_content = "\n".join(getattr(message, "content", "") or "" for message in state["messages"])
    user_query = extract_user_query(all_content)

    monthly_amount = extract_monthly_amount(user_query) or extract_monthly_amount(all_content)
    annual_rate = extract_annual_rate(last_content) or extract_annual_rate(all_content)

    if monthly_amount is None or annual_rate is None:
        repair_message = (
            "[자동 계산 보정]\n"
            "월 납입액 또는 연 금리를 안정적으로 추출하지 못했습니다. "
            "최종 답변에서 만기 수령액 계산이 불가능한 이유를 설명하세요."
        )
    else:
        calculation = calculate_maturity_amount(f"{monthly_amount}, {annual_rate}")
        repair_message = (
            "[자동 계산 보정]\n"
            f"월 납입액: {int(monthly_amount):,}원\n"
            f"연 금리: {annual_rate}%\n"
            f"계산 결과: {calculation}\n\n"
            "이 계산 결과를 사용해 이전 답변을 다시 작성하세요. "
            "'[계산 재시도 필요]' 문구는 최종 답변에 포함하지 마세요."
        )

    return {"messages": [HumanMessage(content=repair_message)]}


def extract_user_query(content: str) -> str:
    marker = "[사용자 질문]"
    if marker not in content:
        return content
    return content.split(marker, 1)[1].split("\n\n", 1)[0].strip()


def build_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("enrich_first_turn", enrich_first_turn)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", call_tools)
    workflow.add_node("repair_calculation", repair_calculation)

    workflow.add_edge(START, "enrich_first_turn")
    workflow.add_edge("enrich_first_turn", "agent")
    workflow.add_conditional_edges(
        "agent",
        route_after_model,
        {"continue": "tools", "repair": "repair_calculation", "end": END},
    )
    workflow.add_edge("tools", "agent")
    workflow.add_edge("repair_calculation", "agent")

    return workflow.compile(checkpointer=MemorySaver())


app = build_graph()
