import streamlit as st
from utils import print_messages, StreamHandler
from langchain_core.messages import ChatMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import os



st.set_page_config(page_title="LLM Chat History", page_icon="😎")
st.title("LLM Chat History 😎")

# API key 설정
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 이전 대화 기록을 출력해주는 코드, utils.py 파일로 분리
print_messages()

def get_session_history(session_ids:str) -> BaseChatMessageHistory:
    # print(session_ids)
    if session_ids not in st.session_state["store"]:
        st.session_state["store"][session_ids] = ChatMessageHistory()
    return st.session_state["store"][session_ids]

# 저장해주기
if "store" not in st.session_state:
    st.session_state["store"] = dict()
    
with st.sidebar:
    session_id = st.text_input("Session ID", value = "abc123")
    
    clear_btn = st.button("대화 기록 초기화")
    if clear_btn:
        st.session_state["messages"] = []
        st.session_state["store"] = dict()
        st.rerun()
  
    

if user_input := st.chat_input("메세지를 입력해주세용"):
    
    st.chat_message("user").write(f"{user_input}")
    # st.session_state["messages"].append(("user", user_input))
    st.session_state["messages"].append(ChatMessage(role="user", content=user_input))
    
    # LLM을 사용해서 AI의 답변을 생성
    # prompt = ChatPromptTemplate.from_template("""질문에 대하여 간결하게 답변해주세요
    #                                 {question}""")
    
    # prompt 코드 가져옴
    
    
    
    # OpenAI Chat API를 사용하여 대화를 생성하는 코드
    with st.chat_message("assistant"):
        # msg = f"당신이 입력한 내용: {user_input}"
        # st.write(f"당신이 입력한 내용: {user_input}")
        stream_handler = StreamHandler(st.empty())
            # 1. 모델 생성
        llm = ChatOpenAI(streaming=True, callbacks=[stream_handler])
        
        
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "질문에 짧고 간결하게 답변해 주세요",
                ),
                
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ]
        )
        
        chain = prompt | llm
        
        chain_with_memory = (
            RunnableWithMessageHistory(
                chain,
                get_session_history,
                input_messages_key="question",
                history_messages_key="history",
            )
        )
        
        
        # chain = prompt | ChatOpenAI()
        # msg = chain.invoke({"question": user_input})
        # response = chain.invoke({"question": user_input})
        
        # 다른 수업에서 가져옴
        response = chain_with_memory.invoke(
            {"question": user_input},
            config = {"configurable": {"session_id": session_id}}
        )
    
        msg = response.content
        # msg = response.content
        # st.write(msg)
        st.session_state["messages"].append(ChatMessage(role="assistant", content=msg))
        

        
    


# 세션 ID를 기반으로 대화 기록을 저장하는 코드

