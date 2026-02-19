import streamlit as st
import tempfile ## for handling temporary files
import os ## for handling file paths

from app.ingestion import VectorStoreManager,split_documents,EmbeddingManager,load_pdfs_from_directory
from app.retriever import HybridRetriever
from app.utils import load_environment
from app.rag_chain import rag_generate
### now we can use streamlit to create a simple interface for our RAG application
def main(): 
    st.title("Satellite Manual RAG Application")
    st.write("Upload satellite manuals in PDF format and ask questions about them!")

    # Step 1: Upload PDFs
    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
    
    if uploaded_files:
        with st.spinner("Processing PDFs..."):
            # Save uploaded files to a temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                
                # Load and process PDFs
                documents = load_pdfs_from_directory(temp_dir) ## load documents from the temporary directory
                split_docs = split_documents(documents) ## split the loaded documents into smaller chunks

                # Initialize embedding manager and vector store manager
                embedding_manager = EmbeddingManager() ## initialize the embedding manager
                vector_store_manager = VectorStoreManager() ## initialize the vector store manager

                # Generate embeddings and store in vector database
                texts = [doc.page_content for doc in split_docs] ## extract text content from the split documents
                embeddings = embedding_manager.generate_embeddings(texts) ## generate embeddings for the extracted texts
                vector_store_manager.collection.add(
                    documents=texts,
                    metadatas=[doc.metadata for doc in split_docs], ## add metadata for each document chunk
                    ids=[str(i) for i in range(len(texts))], ## generate unique IDs for each document chunk
                    embeddings=embeddings.tolist() ## add the generated embeddings to the vector store
                )

                # Initialize hybrid retriever with the processed documents and vector store manager
                hybrid_retriever = HybridRetriever(split_docs, vector_store_manager)

        st.success("PDFs processed successfully! You can now ask questions about the satellite manuals.")

        # Step 2: Ask Questions
        query = st.text_input("Enter your question about the satellite manuals:")
        if query:
            with st.spinner("Generating answer..."):
                answer = rag_generate(query, hybrid_retriever) ## generate an answer using the RAG pipeline
            st.subheader("Answer:")
            st.write(answer)

if __name__ == "__main__":
    load_environment() ## load environment variables
    main() ## run the main function to start the Streamlit app

