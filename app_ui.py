import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from app_agent import app

# 1. 웹 페이지 레이아웃 설정
st.set_page_config(page_title="금융 RAG 에이전트 비서", page_icon="🦜", layout="centered")
st.title("ReAct")
st.caption("66개의 실제 은행 약관 데이터를 바탕으로 추론하고 만기 이자를 계산합니다.")

# 2. 세션 상태를 활용한 대화 기록 메모리 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 3. 기존 대화 내용 화면에 렌더링
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 4. 사용자 질문 입력창 구성
if user_query := st.chat_input("질문할 내용을 입력하세요."):
    
    # 사용자가 입력한 질문 화면에 즉시 표시
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    # 5. 에이전트 호출 및 랭그래프 공식 메모리 세션 연동
    with st.chat_message("assistant"):
        with st.status("금융 에이전트가 약관 분석 및 연산 가동 중...", expanded=False) as status:
            
            # [버그 수정]: app_agent.py의 내장 메모리(MemorySaver)를 쓰기 때문에 과거 메시지를 수동으로 누적해서 채워 보내면 안되고, 
            # 오직 '방금 들어온 신규 질문' 딱 하나만 리스트에 담아 찌름.
            inputs = {"messages": [HumanMessage(content=user_query)]}
            final_answer = ""
            
            # 고유한 thread_id를 넘겨주면, 랭그래프가 알아서 과거 대화 맥락을 메모리에서 복원합니다.
            config = {"configurable": {"thread_id": "sujeong_banking_session_v1"}}
            
            # app.stream 가동
            for output in app.stream(inputs, config=config, stream_mode="updates"):
                for node_name, state_update in output.items():
                    if node_name == "tools":
                        last_msg = state_update["messages"][-1]
                        st.write(f"⚙️ **[Action]** `{last_msg.name}` 도구 연산 가동 완료")
                    
                    if node_name == "agent":
                        last_msg = state_update["messages"][-1]
                        if last_msg.content:
                            final_answer = last_msg.content
            
            status.update(label="✅ 분석 및 자격 조건 검증 완료!", state="complete", expanded=False)

        # =========================================================================
        # [UI 렌더링] 알맹이만 화면에 깔끔하게 출력
        # =========================================================================
        if final_answer:
            st.markdown(final_answer)
            st.session_state.messages.append({"role": "assistant", "content": final_answer})