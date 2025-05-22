import gradio as gr
import os
from dotenv import load_dotenv
import logging
from conversational_rag_chain import conversational_rag_chain
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


def gradio_chat(history, user_input):
    session_id = "gradio-session"  # You can make this dynamic if needed
    try:
        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}},
        )
        answer = response["answer"]
    except Exception as e:
        answer = f"Error: {e}"
    return history + [[user_input, answer]], ""

with gr.Blocks() as demo:
    gr.Markdown("# Conversational RAG Chatbot")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Your question")
    clear = gr.Button("Clear")
    state = gr.State([])

    def user_submit(history, user_input):
        return gradio_chat(history, user_input)

    msg.submit(user_submit, [chatbot, msg], [chatbot, msg])
    clear.click(lambda: ([], ""), None, [chatbot, msg])

demo.launch()