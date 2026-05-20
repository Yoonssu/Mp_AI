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
if user_query := st.chat_input("질문할 내용을 입력하세요. (예: 월 10만원 넣을 건데 이율 좋은 적금 추천해줘)"):
    
    # 사용자가 입력한 질문 화면에 즉시 표시
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    # 5. 에이전트 호출 및 스트리밍 응답 구현
    with st.chat_message("assistant"):
        with st.status("🧠 에이전트가 66개 약관 교차 전수조사 가동 중...", expanded=True) as status:
            
            # [400 에러 해결 핵심] 세션에 쌓인 대화 기록 중 오직 핵심 무결성 대화(User/Assistant 원본 텍스트)만 추출
            graph_messages = []
            for m in st.session_state.messages[:-1]: # 현재 방금 넣은 질문 전까지의 과거 내역 정리
                if m["role"] == "user":
                    graph_messages.append(HumanMessage(content=m["content"]))
                elif m["role"] == "assistant":
                    graph_messages.append(AIMessage(content=m["content"]))
            
            # 방금 들어온 새로운 질문 추가
            graph_messages.append(HumanMessage(content=user_query))
            
            inputs = {"messages": graph_messages}
            final_answer = ""
            
            # LangGraph 실행 및 툴 가동 모니터링 로그 출력
            for output in app.stream(inputs, stream_mode="updates"):
                for node_name, state_update in output.items():
                    if node_name == "tools":
                        last_msg = state_update["messages"][-1]
                        st.write(f"⚙️ **[Action]** `{last_msg.name}` 상세 추론 서랍 오픈 완료")
                    
                    if node_name == "agent":
                        last_msg = state_update["messages"][-1]
                        if last_msg.content:
                            final_answer = last_msg.content
            
            status.update(label="✅ 66개 약관 분석 및 이자율 검증 완료!", state="complete", expanded=False)
        
        # 분석이 완료된 최종 답변을 메인 채팅창에 노출
        st.markdown(final_answer)
        st.session_state.messages.append({"role": "assistant", "content": final_answer})