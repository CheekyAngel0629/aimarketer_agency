# 성공은 했음. 그러나 시간이 지나니 에러가 발생함
# 메모리 문제일 수 있어서 perplexity로 이 부분 개선하려고 함
# 변경한 코드

import streamlit as st
import tiktoken             # text, token간 변환

# 9/26 추가
import gc
import psutil
import asyncio

from loguru import logger   # loguru 라이브러리에서 logger 객체 호출
import os       # 운영체제와 상호작용
import tempfile # 임시 파일 및 임시 디렉터리를 생성하고 관리


from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from langchain_community.llms import Ollama # LLM model

# LangChainDeprecationWarning 개선
from langchain_openai import ChatOpenAI

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader

from langchain_community.document_loaders import TextLoader
# 9/5 txt파일 인식을 위해 추가 (클로드)

from langchain.text_splitter import RecursiveCharacterTextSplitter
# 긴 텍스트를 자연스러운 범위 내에서 분할
from langchain_community.embeddings import HuggingFaceEmbeddings
# Hugging Face의 임베딩 모델을 사용하여 텍스트를 벡터로 변환

from langchain.memory import ConversationBufferMemory
# 대화 메모리 관리. 대화의 기록을 저장, 대화의 문맥을 기억
from langchain_community.vectorstores import FAISS
# 문서나 텍스트 임베한 후, 벡터 저장, 주어진 질문과 가장 유사한 벡터를 빠르게 검색

# 9/26 추가
import faiss
import pickle
import time

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

@st.cache_data(ttl=3600, max_entries=10)  # 1시간 후 만료, 최대 10개 항목 저장
def load_files(data_folder, files_to_load):
        files_text = []
        for filename in files_to_load:
            file_path = os.path.join(data_folder, filename)
            if os.path.exists(file_path):
                files_text.extend(load_document(file_path))
            else:
                st.warning(f"파일을 찾을 수 없습니다: {filename}")
        return files_text

@st.cache_resource(ttl=3600)  # 1시간마다 갱신
def initialize_conversation(_files_text, openai_api_key):
        text_chunks = get_text_chunks(_files_text)
        vetorestore = get_vectorstore(text_chunks)
        return get_conversation_chain(vetorestore, openai_api_key)





def check_resource_usage():
    memory_use = psutil.virtual_memory().percent
    cpu_use = psutil.cpu_percent()
    if memory_use > 80 or cpu_use > 80:
        st.warning(f"High resource usage: Memory {memory_use}%, CPU {cpu_use}%")
    else:
        st.info(f"Current resource usage: Memory {memory_use}%, CPU {cpu_use}%")
    return memory_use, cpu_use


# 메모리 사용량 체크 및 정리 함수
def check_and_clean_memory():
    memory_use = psutil.virtual_memory().percent
    if memory_use > 80:
        gc.collect()
        st.warning(f"High memory usage: {memory_use}%. Cleaning up...")
    return memory_use

# 아래 2개 함수는 9/26 추가
# 메모리 사용량 모니터링 함수
def check_memory_usage():
    memory_use = psutil.virtual_memory().percent
    if memory_use > 80:  # 메모리 사용량이 80%를 넘으면 경고
        st.warning(f"High memory usage: {memory_use}%")
    else:
        st.info(f"Current memory usage: {memory_use}%")
    return memory_use

# 주기적으로 가비지 컬렉션 실행 및 메모리 체크
@st.cache_resource(ttl=300)  # 5분마다 갱신
def periodic_cleanup():
    gc.collect()
    memory_use, cpu_use = check_resource_usage()
    st.session_state['last_memory_check'] = memory_use
    st.session_state['last_cpu_check'] = cpu_use

def auto_cleanup(force=False):
    current_time = time.time()
    if force or ('last_cleanup_time' not in st.session_state) or \
       (current_time - st.session_state.get('last_cleanup_time', 0) > 180):  # 3분마다 체크
        periodic_cleanup()


# 대화 기록 제한 함수
def limit_conversation_history(messages, max_messages=50):
    return messages[-max_messages:]


def main():
    st.set_page_config(page_title="RAG Chat")

    # 메모리 사용량 표시를 위한 placeholder 생성
    memory_placeholder = st.empty()
    
    # 주기적인 메모리 체크 및 정리
    periodic_cleanup()

    # 주기적인 메모리 체크 및 정리
    auto_cleanup()

    # 주기적인 리소스 체크 및 정리
    if 'last_memory_check' not in st.session_state or 'last_cpu_check' not in st.session_state:
        periodic_cleanup()
    

    # 리소스 사용량 업데이트
    memory_use = st.session_state.get('last_memory_check', 0)
    cpu_use = st.session_state.get('last_cpu_check', 0)
    resource_placeholder.info(f"Current resource usage: Memory {memory_use}%, CPU {cpu_use}%")

    # 캐시 지우기
    st.cache_data.clear()
    st.cache_resource.clear()

    st.title("AI마케터 유통업무 Chatbot")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    if 'large_object' in st.session_state:
        del st.session_state['large_object']


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
        
        
                
        # 파일 자동 인식 방식은 여기까지임
                
        # 9/9 버튼 삭제
        # process = st.button("Process")
    
        # 입력받은 환경변수 설정 코드 삭제함

        # 이 아래 부분의 if process를 삭제하고 perplexity가 알려준 코드로 대체
        # 아래 부분을 9/18 재수정함. 기존 코드 삭제하고 새 코드로 변경
        
    
    if "conversation" not in st.session_state:
        if not openai_api_key:
                st.info("Please add all necessary API keys and project information to continue.")
                st.stop()

        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_folder = os.path.join(current_dir, "data")
            

        files_to_load = ["1.개요.docx", "2.매뉴얼.docx"]
        files_text = load_files(data_folder, files_to_load)

        st.session_state.conversation = initialize_conversation(files_text, openai_api_key)
        
    # 9/26 추가
    @st.cache_data(ttl=600)  # 10분 캐시 갱신
    def process_large_data():
        # 큰 데이터 처리 로직
        result = ...
        del large_data  # 처리 후 즉시 삭제
        return result

    # 주기적으로 메모리 체크
    # if st.button("Force Memory Cleanup"):
    #    auto_cleanup(force=True)

    st.session_state.processComplete = True 


    if 'messages' not in st.session_state:
        st.session_state['messages'] = [{"role": "assistant",
                                         "content": "안녕하세요! AI마케터 유통업무 chatbot 입니다. 궁금한 점을 물어보세요."}]
       
    if 'messages' in st.session_state:
        st.session_state['messages'] = limit_conversation_history(st.session_state['messages'])

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):      # avatar를 넣는 등 variation 가능
            st.markdown(message["content"])





    
    def process_user_input(query):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
    
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.conversation({"question": query})
        
            response = result['answer']
            source_documents = result['source_documents']
        
            st.markdown(response)
        
            with st.expander("매뉴얼 참고 내역 확인"):
                for doc in source_documents:
                    st.markdown(doc.metadata['source'], help=doc.page_content)
    
        st.session_state.messages.append({"role": "assistant", "content": response})

        # 대화 처리 후 메모리 정리
        auto_cleanup()

    if query := st.chat_input("유통업무에 대해 물어보세요"):
        process_user_input(query)
        # 대화 처리 후 리소스 체크
        periodic_cleanup()
    
    # 리소스 사용량 업데이트
    memory_use = st.session_state.get('last_memory_check', 0)
    cpu_use = st.session_state.get('last_cpu_check', 0)
    st.info(f"Current resource usage: Memory {memory_use}%, CPU {cpu_use}%")

    # 메모리 사용량 지속적 업데이트
    memory_placeholder.info(f"Current memory usage: {psutil.virtual_memory().percent}%")
        

# 비동기 처리를 위한 함수
async def process_user_input_async(query):
    result = await asyncio.to_thread(st.session_state.conversation, {"question": query})
    return result

# 사용자 입력 처리 함수 수정
def process_user_input(query):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = asyncio.run(process_user_input_async(query))
                response = result['answer']
                source_documents = result['source_documents']
                st.markdown(response)
                with st.expander("참고 문서 확인"):
                    for doc in source_documents:
                        st.markdown(doc.metadata['source'], help=doc.page_content)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    
    # 메모리 체크 및 정리
    check_and_clean_memory()

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



# 클로드가 준 새로운 코드 - 파일 자동 업로드 방식
def load_document(file_path):
    if file_path.endswith('.pdf'):
        return PyPDFLoader(file_path).load_and_split()
    if file_path.endswith('.docx'):
        return Docx2txtLoader(file_path).load_and_split()
    elif file_path.endswith('.pptx'):
        return UnstructuredPowerPointLoader(file_path).load_and_split()
    elif file_path.endswith('.txt'):
        loader = TextLoader(file_path, encoding='utf-8')
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=900,
            chunk_overlap=100,
            length_function=tiktoken_len
        )
        return text_splitter.split_documents(documents)
    else:
        return []  # 지원되지 않는 파일 유형

@st.cache_data
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

@st.cache_resource
def get_vectorstore(_text_chunks):
    """
    주어진 텍스트 청크 리스트로부터 벡터 저장소를 생성합니다.

    이 함수는 Hugging Face의 'jhgan/ko-sroberta-multitask' 모델을 사용하여 각 텍스트 청크의 임베딩을 계산하고,
    이 임베딩들을 FAISS 인덱스에 저장하여 벡터 검색을 위한 저장소를 생성합니다. 이 저장소는 텍스트 청크들 간의
    유사도 검색 등에 사용될 수 있습니다.

    Parameters:
    - text_chunks (List[str]): 임베딩을 생성할 텍스트 청크의 리스트입니다.

    Returns:
    - vectordb (FAISS): 생성된 임베딩들을 저장하고 있는 FAISS 벡터 저장소입니다.

    모델 설명:
    'jhgan/ko-sroberta-multitask'는 문장과 문단을 768차원의 밀집 벡터 공간으로 매핑하는 sentence-transformers 모델입니다.
    클러스터링이나 의미 검색 같은 작업에 사용될 수 있습니다. KorSTS, KorNLI 학습 데이터셋으로 멀티 태스크 학습을 진행한 후,
    KorSTS 평가 데이터셋으로 평가한 결과, Cosine Pearson 점수는 84.77, Cosine Spearman 점수는 85.60 등을 기록했습니다.
"""
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vectordb = FAISS.from_documents(_text_chunks, embeddings)
    return vectordb

@st.cache_resource
def get_conversation_chain(_vetorestore, openai_api_key):
    """
    대화형 검색 체인을 초기화하고 반환합니다.

    이 함수는 주어진 벡터 저장소, OpenAI API 키, 모델 선택을 기반으로 대화형 검색 체인을 생성합니다.
    이 체인은 사용자의 질문에 대한 답변을 생성하는 데 필요한 여러 컴포넌트를 통합합니다.

    Parameters:
    - vetorestore: 검색을 수행할 벡터 저장소입니다. 이는 문서 또는 데이터를 검색하는 데 사용됩니다.
    - openai_api_key (str): OpenAI API를 사용하기 위한 API 키입니다.
    - model_selection (str): 대화 생성에 사용될 언어 모델을 선택합니다. 예: 'gpt-3.5-turbo', 'gpt-4-turbo-preview'.

    Returns:
    - ConversationalRetrievalChain: 초기화된 대화형 검색 체인입니다.

    함수는 다음과 같은 작업을 수행합니다:
    1. ChatOpenAI 클래스를 사용하여 선택된 모델에 대한 언어 모델(LLM) 인스턴스를 생성합니다.
    2. ConversationalRetrievalChain.from_llm 메소드를 사용하여 대화형 검색 체인을 구성합니다. 이 과정에서,
       - 검색(retrieval) 단계에서 사용될 벡터 저장소와 검색 방식
       - 대화 이력을 관리할 메모리 컴포넌트
       - 대화 이력에서 새로운 질문을 생성하는 방법
       - 검색된 문서를 반환할지 여부 등을 설정합니다.
    3. 생성된 대화형 검색 체인을 반환합니다.
    """

    
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-4o-mini", temperature=0)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=_vetorestore.as_retriever(search_type='mmr', verbose=True),
        memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
        get_chat_history=lambda h: h,
        return_source_documents=True,
        verbose=True
    )

    return conversation_chain






if __name__ == '__main__':
    main()