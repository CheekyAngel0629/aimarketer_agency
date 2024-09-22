# 성공은 했음. 그러나 시간이 지나니 에러가 발생함
# 메모리 문제일 수 있어서 perplexity로 이 부분 개선하려고 함
# 변경한 코드

import streamlit as st
import tiktoken             # text, token간 변환

# from loguru import logger   # loguru 라이브러리에서 logger 객체 호출
import os       # 운영체제와 상호작용
import tempfile # 임시 파일 및 임시 디렉터리를 생성하고 관리

# 9/22 추가
import nltk

# 9.23 추가
import asyncio
import hashlib

from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
# from langchain_community.llms import Ollama # LLM model

# from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
# from langchain_community.document_loaders import UnstructuredPowerPointLoader

# 9/5 txt파일 인식을 위해 추가 (클로드)
from langchain_community.document_loaders import TextLoader


# 9/22 마크다운 이해를 위해 추가
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import MarkdownTextSplitter



from langchain.text_splitter import RecursiveCharacterTextSplitter
# 긴 텍스트를 자연스러운 범위 내에서 분할
from langchain_community.embeddings import HuggingFaceEmbeddings
# Hugging Face의 임베딩 모델을 사용하여 텍스트를 벡터로 변환

from langchain.memory import ConversationBufferMemory
# 대화 메모리 관리. 대화의 기록을 저장, 대화의 문맥을 기억
from langchain_community.vectorstores import FAISS
# 문서나 텍스트 임베한 후, 벡터 저장, 주어진 질문과 가장 유사한 벡터를 빠르게 검색

# from streamlit_chat import message
# Streamlit 애플리케이션에서 채팅 인터페이스를 간단하게 구현
# 처음부터 주석처리 되어 있었음

from langchain_community.callbacks import get_openai_callback
# OpenAI API를 호출하는 작업에서 필요한 이벤트를 추적하거나
# 특정 작업이 완료되었을 때 후속 작업을 처리하는 데 사용
from langchain.memory import StreamlitChatMessageHistory
# 채팅 메시지 기록을 관리하는 클래스

# Langsmith api 환경변수 설정 - 아래 .env 추가하면서 주석처리
# os.environ["LANGCHAIN_TRACING_V2"]="true"
# os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"

# .env 파일에 저장된 환경 변수를 파이썬에서 사용할 수 있도록 메모리에 불러옴
# 9/3 코칭시 추가
from dotenv import load_dotenv
load_dotenv()


def main():
    # 캐시 지우기
    st.cache_data.clear()
    st.cache_resource.clear()

    # 9/22 추가
    nltk.download('punkt')

    try:
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
    except Exception as e:
        st.error(f"NLTK 데이터 다운로드 중 오류 발생: {str(e)}")
        return

    st.set_page_config(page_title="RAG Chat")

    st.title("AI마케터 유통업무 Chatbot")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None


    # 9/20 사이드바 및 gpt모델 강제 선택. 원 코드 주석처리 및 새 코드 삽입
    # with st.sidebar:
    #    model_selection = st.selectbox(
    #        "Choose the language model",
    #        ("gpt-4o-mini", "gpt-3.5-turbo", "gpt-4-turbo-preview", "gpt-4o", "bnksys/yanolja-eeve-korean-instruct-10.8b"),
    #        key="model_selection"
    #    )   

    # 9/20 아래 라인 추가    
    st.session_state.model_selection = "gpt-4o-mini"

    openai_api_key = os.getenv("OPEN_API_KEY")
        # api 환경변수, UI 관련 코드는 삭제하였음
        
        # 9/5 파일 업로드 없이 ./data의 파일 자동 업로드 방식 변경하면서 아래 2개 주석 처리
        # uploaded_files = st.file_uploader("Upload your file", type=['pdf', 'docx', 'pptx', 'txt'], accept_multiple_files=True)
        
        # 9/5 ./data의 파일 자동 업로드 방식 변경.
        # txt 파일 업로드 테스트 하려면 아래 주석 처리하고 위를 주석 제거
        
        
        # 9/18 수정. 원 코드는 아예 삭제하고 새로 바꾼
        
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(current_dir, "data")
        
    @st.cache_data(ttl=3600)
    def load_files(data_folder, files_to_load):
        files_text = []
        for filename in files_to_load:
            file_path = os.path.join(data_folder, filename)
            if os.path.exists(file_path):
                files_text.extend(load_document(file_path))
            else:
                st.warning(f"파일을 찾을 수 없습니다: {filename}")
        return files_text

    @st.cache_resource(ttl=3600)
    def initialize_conversation(_files_text, openai_api_key):
        text_chunks = get_text_chunks(_files_text)
        vectorstore = get_vectorstore(text_chunks)
        return get_conversation_chain(vectorstore, openai_api_key)

    # 합본 파일로 변경
    files_to_load = ["대리점_매뉴얼.md"]
    files_text = load_files(data_folder, files_to_load)
        
                
        # 파일 자동 인식 방식은 여기까지임
                
        # 9/9 버튼 삭제
        # process = st.button("Process")
    
        # 입력받은 환경변수 설정 코드 삭제함

        # 이 아래 부분의 if process를 삭제하고 perplexity가 알려준 코드로 대체
        # 아래 부분을 9/18 재수정함. 기존 코드 삭제하고 새 코드로 변경
    @st.cache_resource(ttl=3600)
        
    def initialize_conversation(_files_text, openai_api_key):
            text_chunks = get_text_chunks(_files_text)
            vetorestore = get_vectorstore(text_chunks)
            return get_conversation_chain(vetorestore, openai_api_key)
    if "conversation" not in st.session_state:
            if not openai_api_key:
                    st.info("Please add all necessary API keys and project information to continue.")
                    st.stop()
    
    st.session_state.conversation = initialize_conversation(files_text, openai_api_key)
    st.session_state.processComplete = True 




    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant",
                                         "content": "안녕하세요! AI마케터 유통업무 chatbot 입니다. 궁금한 점을 물어보세요."}]
        

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # history = StreamlitChatMessageHistory(key="chat_messages")
    # 원래 주석

    # Chat logic
    # 9/18 변경
    
    async def process_user_input(query):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = await asyncio.to_thread(st.session_state.conversation, {"question": query})
                response = result['answer']
                source_documents = result['source_documents']
            st.markdown(response)
            with st.expander("참고 문서 확인"):
                for doc in source_documents:
                    st.markdown(doc.metadata['source'], help=doc.page_content)
        st.session_state.messages.append({"role": "assistant", "content": response})

    if query := st.chat_input("Message to chatbot"):
        process_user_input(query)
        
        
def tiktoken_len(text):
    """
    주어진 텍스트에 대한 토큰 길이를 계산합니다.

    Parameters:
    - text: str, 토큰 길이를 계산할 텍스트입니다.

    Returns:
    - int, 계산된 토큰 길이입니다.
    """
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)



# 파일 자동 업로드 방식으로 변경 (클로드)
def load_document(file_path):
    if file_path.endswith('.docx'):
        return Docx2txtLoader(file_path).load_and_split()
    # 마크다운 양식 읽도록 변경 - 9/22
    # elif file_path.endswith('.docx'):
    #    loader = Docx2txtLoader(file_path)
    #    documents = loader.load()
    #    markdown_splitter = MarkdownTextSplitter(chunk_size=900, chunk_overlap=100)
    #    return markdown_splitter.split_documents(documents)
    # elif file_path.endswith('.pptx'):
    #     return UnstructuredPowerPointLoader(file_path).load_and_split()
    elif file_path.endswith('.txt'):
        loader = TextLoader(file_path, encoding='utf-8')
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=900,
            chunk_overlap=100,
            length_function=tiktoken_len
        )
        return text_splitter.split_documents(documents)
    # 마크다운 확장자 추가 - 9/22
    # elif file_path.endswith('.md'):
    #    loader = UnstructuredMarkdownLoader(file_path)
    #    documents = loader.load()
    #    markdown_splitter = MarkdownTextSplitter(chunk_size=900, chunk_overlap=100)
    #    return markdown_splitter.split_documents(documents)
    elif file_path.endswith('.md'):
        loader = TextLoader(file_path, encoding='utf-8')
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=100)
        return text_splitter.split_documents(documents)

    else:
        return []  # 지원되지 않는 파일 유형


def get_text(docs):
    doc_list = []
    for doc in docs:
        doc_list.extend(load_document(doc))
    return doc_list


def get_text_chunks(text):
    """
    주어진 텍스트 목록을 특정 크기의 청크로 분할합니다.

    이 함수는 'RecursiveCharacterTextSplitter'를 사용하여 텍스트를 청크로 분할합니다. 각 청크의 크기는
    `chunk_size`에 의해 결정되며, 청크 간의 겹침은 `chunk_overlap`으로 조절됩니다. `length_function`은
    청크의 실제 길이를 계산하는 데 사용되는 함수입니다. 이 경우, `tiktoken_len` 함수가 사용되어 각 청크의
    토큰 길이를 계산합니다.

    Parameters:
    - text (List[str]): 분할할 텍스트 목록입니다.

    Returns:
    - List[str]: 분할된 텍스트 청크의 리스트입니다.

    사용 예시:
    텍스트 목록이 주어졌을 때, 이 함수를 호출하여 각 텍스트를 지정된 크기의 청크로 분할할 수 있습니다.
    이렇게 분할된 청크들은 텍스트 분석, 임베딩 생성, 또는 기계 학습 모델의 입력으로 사용될 수 있습니다.


    주의:
    `chunk_size`와 `chunk_overlap`은 분할의 세밀함과 처리할 텍스트의 양에 따라 조절할 수 있습니다.
    너무 작은 `chunk_size`는 처리할 청크의 수를 불필요하게 증가시킬 수 있고, 너무 큰 `chunk_size`는
    메모리 문제를 일으킬 수 있습니다. 적절한 값을 실험을 통해 결정하는 것이 좋습니다.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    chunks = text_splitter.split_documents(text)
    return chunks


def hash_vectorstore(vectorstore):
    return hashlib.md5(str(id(vectorstore)).encode()).hexdigest()

@st.cache_data(hash_funcs={FAISS: hash_vectorstore})
def get_vectorstore(_text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    return vectordb


def get_conversation_chain(vetorestore, openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-4o-mini", temperature=0)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vetorestore.as_retriever(search_type='mmr', search_kwargs={"k": 3}),
        memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
        get_chat_history=lambda h: h,
        return_source_documents=True,
        verbose=False
    )
    return conversation_chain



if __name__ == '__main__':
    main()