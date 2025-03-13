import streamlit as st
from utils import print_messages, StreamHandler
from langchain.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_core.messages import ChatMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain import hub
import os

# 페이지 설정
st.set_page_config(page_title="논문 대신 읽어줌", page_icon="😎")
st.title("논문 대신 읽어줌 😎")

# API key 설정
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"] 

# 대화 기록을 저장하는 코드
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 이전 대화 기록을 출력해주는 코드, utils.py 파일로 분리
print_messages()

def get_session_history(session_ids:str) -> BaseChatMessageHistory:
    if session_ids not in st.session_state["store"]:
        st.session_state["store"][session_ids] = ChatMessageHistory()
    return st.session_state["store"][session_ids]

def format_docs(docs):
    # 검색한 문서 결과를 하나의 문단으로 합쳐줍니다.
    return "\n\n".join(doc.page_content for doc in docs)

# 저장해주기
if "store" not in st.session_state:
    st.session_state["store"] = dict()

# 세션 초기화 및 세션 ID 입력
with st.sidebar:
    session_id = st.text_input("Session ID", value = "abc123")
    
    clear_btn = st.button("대화 기록 초기화")
    if clear_btn:
        st.session_state["messages"] = []
        st.session_state["store"] = dict()
        st.rerun()

# PDF 파일 업로드
uploaded_file = st.file_uploader("논문 PDF 파일을 업로드하세요", type=["pdf"])

# PDF 파일이 업로드 되었을 때
if uploaded_file is not None:
    with st.spinner("PDF 처리 중..."):
        # 파일 저장
        file_path = f"temp_{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # PDF 로드
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        # 텍스트 청크 분할
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        split_docs = text_splitter.split_documents(docs)

        # 벡터스토어 생성
        vectorstore = FAISS.from_documents(split_docs, OpenAIEmbeddings())
        
        # 벡터스토어를 검색기로 변환
        retriever = vectorstore.as_retriever()

        st.success("PDF 업로드 및 벡터 저장 완료!")

  
# 사용자 입력을 받아서 대화를 생성하는 코드  
if user_input := st.chat_input("메세지를 입력해주세용"):
    
    st.chat_message("user").write(f"{user_input}")
    st.session_state["messages"].append(ChatMessage(role="user", content=user_input))
    
    
    # OpenAI Chat API를 사용하여 대화를 생성하는 코드
    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())
        
        # 1. 모델 생성
        llm = ChatOpenAI(model_name="gpt-4o", temperature = 0, streaming=True, callbacks=[stream_handler])
        
        # 2. 대화 생성
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "질문에 짧고 간결하게 답변해 주세요",
                ),
                
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),
            ]
        )
        
        prompt = hub.pull("rlm/rag-prompt")
          
        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | qa_prompt 
            | llm
                 )
        
        # 3. 대화 생성, 메모리 기능 추가, 스트림 출력
        chain_with_memory = (
            RunnableWithMessageHistory(
                chain,
                get_session_history,
                input_messages_key="question",
                history_messages_key="history",
            )
        )
        
        # 4. 답변 생성
        response = chain_with_memory.invoke(
            {"question": user_input},
            config = {"configurable": {"session_id": session_id}}
        )

        # 5. 답변 출력
        msg = response.content
        
        # 6. 대화 기록 저장
        st.session_state["messages"].append(ChatMessage(role="assistant", content=msg))
