import streamlit as st
from langchain_core.messages.chat import ChatMessage
from rag.pdf import PDFRetrievalChain
from langchain_teddynote import logging
from rag.evaluation import RagEvaluator
from dotenv import load_dotenv
import os

# API KEY 정보로드
load_dotenv()

# 프로젝트 로그
logging.langsmith("[Project] PDF paper RAG With Evaluation")

# 캐시 디렉토리 생성
os.makedirs(".cache/files", exist_ok=True)
os.makedirs(".cache/embeddings", exist_ok=True)

st.set_page_config(page_title="논문 봇 ✅", page_icon="✅")
st.title("논문 봇 ✅")

# 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "chain" not in st.session_state:
    st.session_state["chain"] = None

if "evaluator" not in st.session_state:
    st.session_state["evaluator"] = RagEvaluator()

if "expected_question" not in st.session_state:
    st.session_state["expected_question"] = None

# 사이드바
with st.sidebar:
    clear_btn = st.button("대화 초기화")
    uploaded_file = st.file_uploader("파일 업로드", type=["pdf"])
    eval_toggle = st.toggle("평가 결과 출력", value=True)
    
    if st.button("결과 출력", key="eval_all", type="primary", use_container_width=True):
        evaluator = st.session_state["evaluator"]
        if len(evaluator.get_samples()["question"]) > 0:
            with st.spinner("평가 중입니다."):
                eval_df = evaluator.evaluate_all()
                result_df = eval_df[["faithfulness", "answer_relevancy"]].mean()
                result_df.name = "평균 점수"
                st.dataframe(result_df, use_container_width=True)
        else:
            st.error("평가할 데이터가 없습니다.")

# 파일을 캐시 저장
@st.cache_resource(show_spinner="업로드한 파일을 처리 중입니다...")
def embed_file(file):
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file.read())
    return file_path

# 체인 생성
def create_rag_chain(file_path):
    pdf = PDFRetrievalChain([file_path]).create_chain()
    return pdf.retriever, pdf.chain

# 예상 질문 생성
def generate_expected_question(user_input):
    chain = st.session_state["chain"]
    retriever = st.session_state["retriever"]
    context = retriever.invoke(user_input)  # 문서에서 관련 내용 찾기

    # 예상 질문 생성 요청
    response = chain.invoke({"question": user_input, "context": context, "mode": "generate_question"})
    return response.strip()

# 파일 업로드 시 체인 생성
if uploaded_file:
    file_path = embed_file(uploaded_file)
    retriever, chain = create_rag_chain(file_path)
    st.session_state["retriever"] = retriever
    st.session_state["chain"] = chain

# 초기화 버튼
if clear_btn:
    st.session_state["messages"] = []
    st.session_state["evaluator"] = RagEvaluator()
    st.session_state["expected_question"] = None

# 이전 대화 출력
for chat_message in st.session_state["messages"]:
    st.chat_message(chat_message.role).write(chat_message.content)

# 사용자 입력 받기
user_input = st.chat_input("궁금한 내용을 물어보세요!")

if user_input:
    chain = st.session_state["chain"]
    retriever = st.session_state["retriever"]
    evaluator = st.session_state["evaluator"]

    if chain is not None and retriever is not None:
        # 사용자 질문 처리
        st.chat_message("user").write(user_input)

        if st.session_state["expected_question"] is None:
            # 첫 질문이므로 예상 질문 생성
            expected_question = generate_expected_question(user_input)
            st.session_state["expected_question"] = expected_question
            st.chat_message("assistant").write(f"이와 관련된 질문을 해볼게요: **{expected_question}**")
        else:
            # 사용자가 예상 질문에 답변하면 정답 제공
            user_answer = user_input
            expected_question = st.session_state["expected_question"]

            # 문서에서 정답 찾기
            context = retriever.invoke(expected_question)
            response = chain.invoke({"question": expected_question, "context": context})
            
            # 디버깅용 출력
            st.write(f"⏳ [DEBUG] response: {response}")

            if response is None:
                response = "답변을 찾을 수 없습니다."

            st.chat_message("assistant").write(f"✅ 정답: {response.strip()}")

            # 새로운 예상 질문 생성
            new_expected_question = generate_expected_question(expected_question)
            st.session_state["expected_question"] = new_expected_question
            st.chat_message("assistant").write(f"다음 질문: **{new_expected_question}**")

            # RAGAS 평가 추가
            evaluator.add_sample(expected_question, response, context)
            if eval_toggle:
                with st.spinner("평가 중입니다."):
                    eval_result = evaluator.evaluate_last()
                st.chat_message("assistant").write(
                    f'✅ 평가 결과\n- 관련성 점수: {eval_result.iloc[0]["answer_relevancy"]:.3f}\n- 신뢰도 점수: {eval_result.iloc[0]["faithfulness"]:.3f}'
                )

        # 대화 기록 저장
        st.session_state["messages"].append(ChatMessage(role="user", content=user_input))
        st.session_state["messages"].append(ChatMessage(role="assistant", content=response.strip()))

    else:
        st.error("파일을 업로드 해주세요.")
