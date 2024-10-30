import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from datasets import load_dataset
from langchain.docstore.document import Document as LangchainDocument
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_huggingface import HuggingFaceEmbeddings

class RagModel:
  def __init__(self, model_name, google_api_key):
    print("RAG model initializing...")
    self._llm = ChatGoogleGenerativeAI(model=model_name, top_p=1, temperature=0.001, google_api_key=google_api_key)
    self._chat_history = {}
    self._ds = load_dataset("Luke3501/pcb_practice", split="train", token=os.environ["HF_WRITE"], cache_dir="/pcb_defect_detect/cache")
    self._vector_db = self._init_vector_db()
    self._retriever = self._get_retriever()
    print("RAG model initialized")
    
  def _get_memory(self, session_id="foobar-default") -> BaseChatMessageHistory:
    """Returns memory of the session

    :param session_id: The keyname is magic unless override `history_factory_config`
                       in `RunnableWithMessageHistory`
    :return:
    """
    if session_id not in self._chat_history:
        self._chat_history[session_id] = ChatMessageHistory()
    return self._chat_history[session_id]
  
  def _init_vector_db(self):
    docs_processed = self._format_docs()
    EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
    embedding_model = HuggingFaceEmbeddings(
      model_name=EMBEDDING_MODEL_NAME,
      model_kwargs={"device": "cpu"},
      encode_kwargs={"normalize_embeddings": True},  # set True for cosine similarity
      cache_folder="/pcb_defect_detect/cache"
    )

    # 创建知识向量数据库
    return FAISS.from_documents(
      docs_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE
    )
    
  def _get_retriever(self):
    vector_store = self._vector_db
    retriever = vector_store.as_retriever(search_kwargs={"k": 50})
    return retriever
  
  def _format_docs(self):
    MARKDOWN_SEPARATORS = [
      "\n#{1,6} ",
      "```\n",
      "\n\\*\\*\\*+\n",
      "\n---+\n",
      "\n___+\n",
      "\n\n",
      "\n",
      " ",
      "",
    ]

    text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=600,
      chunk_overlap=100,
      add_start_index=True,
      strip_whitespace=True,
      separators=MARKDOWN_SEPARATORS,
    )

    RAW_KNOWLEDGE_BASE = [
      LangchainDocument(page_content=doc["text"], metadata={"source": doc["source"]})
      for doc in self._ds
    ]

    docs_processed = []
    for doc in RAW_KNOWLEDGE_BASE:
      docs_processed += text_splitter.split_documents([doc])

    return docs_processed
  
  def query(self, question, session_id="foobar-default"):
    CONDENSE_PROMPT = """Given a chat history and the latest user question
    which might reference context in the chat history, formulate a standalone question 
    which can be understood without the chat history. Do NOT answer the question, 
    just reformulate it if needed and otherwise return it as is.
    """

    QA_PROMPT = "You are an assistant for question-answering tasks. "\
            "Use the following pieces of retrieved context to answer "\
            "the question. The questions will be mostly about PCB defects. Try"\
            "to analyze the root cause of PCB defects based on following aspects --"\
            "temparature, pressure, base material, production process, and etc."\
            "If the question is not PCB-related, say that you don't know."\
            "If you don't know the answer, say that you "\
            "don't know. Keep the answer concise. "\
            "\n\n"\
            "{context}"
    
    retriever = self._get_retriever()
    llm = self._llm
    chat_history = self._get_memory(session_id)

    # 更新聊天歷史
    chat_history.add_user_message(question)

    # 使用 LCEL 語法設置聊天歷史的上下文
    response = (
      create_retrieval_chain(
        create_history_aware_retriever(
          llm, retriever, ChatPromptTemplate.from_messages([
            ("system", CONDENSE_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
          ])
        ),
        create_stuff_documents_chain(
          llm, ChatPromptTemplate.from_messages([
            ("system", QA_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
          ])
        )
      )
      .invoke(
        input={"input": question, "chat_history": chat_history.messages},
        config={"configurable": {"session_id": session_id}}
      )
    )

    chat_history.add_ai_message(response["answer"])
    return response["answer"]