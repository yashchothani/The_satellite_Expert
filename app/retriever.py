from rank_bm25 import BM25Okapi ## for BM25 retrieval
from typing import List, Any, Dict, Tuple
import numpy as np
from app.ingestion import split_documents,EmbeddingManager,VectorStoreManager ## import necessary components from the ingestion module
"""🛰️ PHASE 5 — Hybrid Retrieval (Important)
Satellite manuals contain terms like:
Apogee
Perigee
Conjunction
Delta-V
GEO transfer
Sun-synchronous orbit
Vector search sometimes fails for rare terms.
So we combine:
1️⃣ Vector Similarity
2️⃣ BM25 Keyword Search
This hybrid approach ensures we retrieve relevant sections even if embeddings struggle with niche terms."""

class HybridRetriever:
    def __init__(self,documents:List[Any],vector_store_manager:VectorStoreManager):
        """Initialize the hybrid retriever with documents and vector store manager.
        Args:
            documents (List[Any]): A list of documents to be used for BM25 retrieval.
            vector_store_manager (VectorStoreManager): An instance of the VectorStoreManager for vector retrieval.
        """
        self.documents=documents ## store the documents for BM25 retrieval
        self.vector_store_manager=vector_store_manager ## store the vector store manager for vector retrieval
        self.tokenized_corpus=[
            doc.page_content.split() for doc in documents
        ]## tokenize the document contents for BM25 retrieval
        self.bm25=BM25Okapi(self.tokenized_corpus) ## initialize the BM25 retriever with the tokenized corpus
    def retrieve(self,query:str,top_k:int=5)->List[Tuple[str,Dict[str,Any]]]:
        """Retrieve relevant documents based on a query using a hybrid approach of vector similarity and BM25 keyword search.
        Args:
            query (str): The input query for retrieval.
            top_k (int): The number of top results to return from each retrieval method.
        Returns:
            List[Tuple[str, Dict[str, Any]]]: A list of tuples containing the retrieved document content and its metadata.
        """
        # Step 1: Vector Similarity Retrieval
        vector_results=self.vector_store_manager.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        vector_retrieved_docs=[
            (doc, metadata) for doc, metadata in zip(vector_results["documents"][0], vector_results["metadatas"][0])
        ]

        # Step 2: BM25 Keyword Search
        tokenized_query=query.split() ## tokenize the query for BM25 retrieval
        bm25_scores=self.bm25.get_scores(tokenized_query) ## get BM25 scores for the query against the corpus
        top_bm25_indices=np.argsort(bm25_scores)[::-1][:top_k] ## get indices of top BM25 results

        bm25_retrieved_docs=[
            (self.documents[idx].page_content, self.documents[idx].metadata) for idx in top_bm25_indices
        ]

        # Combine and deduplicate results
        combined_results=vector_retrieved_docs + bm25_retrieved_docs ## combine results from both methods
        seen=set()
        unique_results=[]
        for doc, metadata in combined_results:
            if doc not in seen:
                unique_results.append((doc, metadata))
                seen.add(doc)

        return unique_results[:top_k] ## return top_k unique results