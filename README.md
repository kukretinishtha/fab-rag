# 🧠 Conversational RAG Chatbot with LangChain, FAISS, and Gradio

A context-aware chatbot using Retrieval-Augmented Generation (RAG), built with:

- 🔗 **LangChain** for orchestration
- 🧠 **FAISS** for fast vector search
- 🗣️ **OpenAI's GPT models** for intelligent responses
- 🎛️ **Gradio** for the interactive web UI

---

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20HuggingFace-Demo-blueviolet?logo=huggingface&logoColor=white)](https://huggingface.co/spaces/Nishthaaa/langchain_openai_rag_chatbot)

### 🔗 [Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/Nishthaaa/langchain_openai_rag_chatbot)

---

## 📁 Features

- ✅ Loads `.txt` documents from a local folder
- ✅ Builds or loads a FAISS index for fast semantic search
- ✅ Uses `SentenceTransformer` for embeddings
- ✅ GPT-based Q&A using only retrieved context
- ✅ Follows conversation history for follow-up questions
- ✅ Web UI built with Gradio and hosted on Hugging Face

---

## 🚀 Getting Started (Local Setup)

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/conversational-rag-chatbot.git
cd conversational-rag-chatbot
````

### 2. Create and Activate a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 3. Install Required Packages

```bash
pip install -r requirements.txt
```

### 4. Create a `.env` File

```env
FOLDER_PATH=data/ancient_greece
INDEX_PATH=faiss_index
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
LLM_MODEL_NAME=gpt-4o-mini
```

Put your `.txt` files in the `data/ancient_greece/` directory.

### 5. Run the Application

```bash
python app.py
```

This will launch a Gradio interface at `http://localhost:7860`.

---

## 📂 Folder Structure

```
.
├── app.py                     # Main chatbot script
├── ancient_greece/        # Text documents go here
├── faiss_index/               # Auto-generated FAISS index
├── .env                       # Environment variables
├── requirements.txt
└── README.md
```

---

## 🧠 How It Works

1. **Ingestion**: `.txt` files are loaded and cleaned.
2. **Embedding**: Text is embedded using `SentenceTransformer`.
3. **Indexing**: Vectors are stored in a FAISS index for similarity search.
4. **Conversation Flow**:

   * Follow-up questions are rewritten into standalone ones.
   * Contextually relevant documents are retrieved.
   * GPT responds using only the retrieved content.
5. **Session History**: Tracked using LangChain’s message history interface.

---

## 💬 Example Interactions

> **User**: Who was Socrates?
> **Bot**: Socrates was a classical Greek philosopher known for...

> **User**: What were his beliefs about virtue?
> **Bot**: Socrates believed that virtue is a kind of knowledge...

> **User**: And what did Plato say about that?
> **Bot**: Plato expanded on Socrates’ ideas, stating that...

---

## ☁️ Hosted Version

You can try this app live on Hugging Face Spaces:

👉 [**Click here for the demo**](https://huggingface.co/spaces/Nishthaaa/langchain_openai_rag_chatbot)

---

## 🛡️ Security Note

> ⚠️ FAISS index is loaded with `allow_dangerous_deserialization=True`.
> Only load trusted indexes to avoid potential security risks.

---

## 🛠️ Deployment on Hugging Face

To deploy your own version:

1. Push your app code and `.env` to secret section of Hugging Face under space setiing.
2. Ensure `app.py` is the entry point.
3. Add `requirements.txt` with all dependencies.

---

## 📈 Future Improvements

* 🔄 Support for PDFs, DOCX, or Markdown files
* 🧾 Downloadable chat history
* 🗂️ File upload and auto-indexing
* 🔐 User session tracking via cookies or tokens
* 🎨 UI customization and themes

---

## 🙌 Acknowledgements

* [LangChain](https://www.langchain.com/)
* [OpenAI](https://platform.openai.com/)
* [Gradio](https://gradio.app/)
* [FAISS](https://github.com/facebookresearch/faiss)
* [SentenceTransformers](https://www.sbert.net/)