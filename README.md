# 🛰️ The Satellite Expert - RAG Application

A powerful Retrieval-Augmented Generation (RAG) application designed to answer questions from satellite manuals and technical documents. Built with **Streamlit**, **LangChain**, and **Groq**.

## 🚀 Features

- **Document Ingestion**: Upload and process multiple PDF manuals.
- **Hybrid Retrieval**: Combines **Vector Similarity Search** (ChromaDB + Sentence Transformers) and **Keyword Search** (BM25) for accurate results.
- **Advanced RAG Pipeline**: Uses **Groq's Llama 3** model for high-speed, intelligent responses.
- **Interactive UI**: Simple and intuitive interface built with Streamlit.
- **Source Citations**: Retrieves and displays relevant document chunks and metadata.

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **LLM**: [Groq](https://groq.com/) (Llama-3-70b-versatile)
- **Orchestration**: [LangChain](https://www.langchain.com/)
- **Vector Database**: [ChromaDB](https://www.trychroma.com/)
- **Embeddings**: [SentenceTransformers](https://www.sbert.net/) (`all-MiniLM-L6-v2`)
- **Keyword Search**: [Rank-BM25](https://github.com/dorianbrown/rank_bm25)

## 📂 Project Structure

```bash
The_Satellite_Expert/
├── app/
│   ├── ingestion.py       # PDF loading, splitting, embedding, and vector store management
│   ├── retriever.py       # HybridRetriever class (Vector + BM25)
│   ├── rag_chain.py       # RAG pipeline definition and execution
│   └── utils.py           # Utility functions (e.g., environment loading)
├── data/
│   └── vectorStore/       # Persisted ChromaDB data
├── streamlit_app.py       # Main application entry point
├── requirements.txt       # Project dependencies
├── .env                   # Environment variables (API keys)
└── README.md              # Project documentation
```

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd The_Satellite_Expert
```

### 2. Create a Virtual Environment
It's recommended to use a virtual environment to manage dependencies.
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the root directory and add your Groq API key:
```ini
GROQ_API_KEY=your_groq_api_key_here
```

## ▶️ Usage

Run the Streamlit application:
```bash
streamlit run streamlit_app.py
```

1.  **Upload PDFs**: Use the file uploader in the sidebar/main area to upload your satellite manuals.
2.  **Process**: The app will digest the documents, creating embeddings and indexes.
3.  **Ask Questions**: Type your query in the text input box (e.g., *"What is the procedure for orbital insertion?"*).
4.  **Get Answers**: The AI will generate a response based on the uploaded content.

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.
