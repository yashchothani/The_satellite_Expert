###simple rag pipeline with groq llm
from langchain_groq import ChatGroq ## for using Groq LLM in a chat format
import os
from pathlib import Path
from app.retriever import HybridRetriever ## import the HybridRetriever class from the retriever module
from dotenv import load_dotenv ## for loading environment variables from a .env file

# Load environment variables from .env file
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable not found. Please check your .env file.")

llm=ChatGroq(api_key=groq_api_key,temperature=0.1,model_name="llama-3.3-70b-versatile",max_tokens=1024)### initialize the Groq LLM with specified parameters    

### function to perform retrieval augmented generation (RAG) using the hybrid retriever and Groq LLM
def rag_generate(query:str,hybrid_retriever:HybridRetriever,llm:ChatGroq=llm,top_k:int=5)->str:
    """Perform retrieval augmented generation (RAG) using the hybrid retriever and Groq LLM.
    Args:
        query (str): The input query for RAG.
        hybrid_retriever (HybridRetriever): An instance of the HybridRetriever for retrieving relevant documents.
        llm (ChatGroq): An instance of the ChatGroq LLM for generating responses.
        top_k (int): The number of top results to retrieve and use for generation.
    Returns:
        str: The generated response from the LLM based on the retrieved documents and the query.
    """
    retrieved_docs=hybrid_retriever.retrieve(query, top_k=top_k) ## retrieve relevant documents using the hybrid retriever
    context="\n\n".join([f"Document:\n{doc}\nMetadata:\n{metadata}" for doc, metadata in retrieved_docs]) ## create a context string from the retrieved documents and their metadata
    prompt=f"Use the following retrieved documents to answer the question:\n\n{context}\n\nQuestion: {query}\nAnswer:" ## create a prompt for the LLM that includes the context and the query
    response=llm.invoke(prompt) ## generate a response from the LLM based on the prompt
    return response.content ## return the content of the generated response


