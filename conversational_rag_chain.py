import gradio as gr
import os
import re
from pathlib import Path
from dotenv import load_dotenv

from langchain.schema import Document
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import logging

# --- Setup logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Load environment variables ---
load_dotenv()
# --- Configurable paths and model names ---
FOLDER_PATH = os.getenv("FOLDER_PATH", "data/ancient_greece")
INDEX_PATH = os.getenv("INDEX_PATH", "faiss_index")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")

# --- Utility functions ---
def load_documents_from_folder(folder_path: str):
    """Load each .txt file in the folder as a separate Document."""
    folder = Path(folder_path)
    if not folder.exists():
        logger.error(f"Folder not found: {folder_path}")
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    txt_files = list(folder.glob("*.txt"))
    documents = []
    for file in txt_files:
        try:
            text = file.read_text(encoding="utf-8")
            clean_text = re.sub(r'\s+', ' ', text.strip())
            doc = Document(page_content=clean_text, metadata={"source": file.name})
            documents.append(doc)
        except Exception as e:
            logger.warning(f"Error reading {file}: {e}")
    if not documents:
        logger.error(f"No .txt documents found in {folder_path}")
        raise ValueError(f"No .txt documents found in {folder_path}")
    logger.info(f"Loaded {len(documents)} documents from {folder_path}")
    return documents

def build_or_load_faiss_index(documents, embedding_model, index_path):
    """Builds or loads a FAISS index from disk."""
    if Path(index_path).exists():
        logger.info(f"Loading FAISS index from {index_path}")
        return FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)
    else:
        logger.info("Building new FAISS index...")
        vectorstore = FAISS.from_documents(documents, embedding_model)
        vectorstore.save_local(index_path)
        logger.info(f"Saved FAISS index to {index_path}")
        return vectorstore

def get_session_history(store, session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# --- Pipeline setup ---
try:
    documents = load_documents_from_folder(FOLDER_PATH)
    # embedding_model = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    embedding_model = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectorstore = build_or_load_faiss_index(documents, embedding_model, INDEX_PATH)
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(model=LLM_MODEL_NAME)
except Exception as e:
    logger.error(f"Pipeline setup failed: {e}")
    raise

# --- Prompt templates ---
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that rewrites follow-up questions into standalone questions using chat history."),
    MessagesPlaceholder("chat_history"),
    ("human", """Given the above chat history and the latest user question below,
reformulate it into a standalone question. Do not answer the question.
If it's already standalone, return it as is.

Latest user question:
{input}"""),
])

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an assistant for question-answering tasks.
Answer this question using the provided context only.
If you don't know the answer, just say 'I don't know'
{context}"""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
contextual_rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# --- Session store and conversational chain ---
store = {}

conversational_rag_chain = RunnableWithMessageHistory(
    contextual_rag_chain,
    lambda session_id: get_session_history(store, session_id),
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# --- Example usage (can be removed in production) ---
def ask_question(question, session_id):
    try:
        response = conversational_rag_chain.invoke(
            {"input": question},
            config={"configurable": {"session_id": session_id}},
        )
        return response["answer"]
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        return "An error occurred during question answering."