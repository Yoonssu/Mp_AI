import streamlit as st
from langchain_core.messages import HumanMessage

from app.agent.graph import app


st.set_page_config(page_title="금융 RAG 에이전트", page_icon="💬", layout="centered")
st.title("금융 RAG 에이전트")
st.caption("은행 약관 데이터를 검색하고, 우대 조건과 만기 예상액을 함께 분석합니다.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_query := st.chat_input("예: 월 30만원씩 넣을 수 있는 청년 적금 중 추천해줘"):
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("assistant"):
        with st.status("약관 검색과 계산을 진행하는 중...", expanded=False) as status:
            inputs = {"messages": [HumanMessage(content=user_query)]}
            config = {"configurable": {"thread_id": "banking_session_v1"}}
            final_answer = ""

            for output in app.stream(inputs, config=config, stream_mode="updates"):
                for node_name, state_update in output.items():
                    if node_name == "tools":
                        last_msg = state_update["messages"][-1]
                        st.write(f"**도구 실행 완료:** `{last_msg.name}`")
                    elif node_name == "agent":
                        last_msg = state_update["messages"][-1]
                        if last_msg.content:
                            final_answer = last_msg.content

            status.update(label="분석 완료", state="complete", expanded=False)

        if final_answer:
            st.markdown(final_answer)
            st.session_state.messages.append({"role": "assistant", "content": final_answer})
        else:
            error_message = "응답을 생성하지 못했습니다. API 키와 ChromaDB 데이터를 확인해 주세요."
            st.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})
