"""
Hybrid search combining vector similarity with keyword (BM25) matching.
Uses Reciprocal Rank Fusion (RRF) to combine results.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi
import re


class HybridSearch:
    """
    Combine vector search results with BM25 keyword scores using Reciprocal Rank Fusion (RRF).
    
    This enables better search quality by leveraging both semantic (vector) 
    and lexical (keyword) matching.
    """

    def __init__(
        self,
        vector_index,
        text_corpus: List[str],
        tokenizer_pattern: str = r'\w+',
        k_rrf: int = 60
    ):
        """
        Args:
            vector_index: A search object with a .search(query_embedding, k) method.
                         (e.g., FAISSSearch or MilvusSearch)
            text_corpus: List of text strings (e.g., track titles/descriptions) for BM25.
            tokenizer_pattern: regex pattern for tokenization.
            k_rrf: RRF constant (usually 60).
        """
        self.vector_index = vector_index
        self.text_corpus = text_corpus
        self.tokenizer_pattern = tokenizer_pattern
        self.k_rrf = k_rrf

        # Tokenize corpus for BM25
        tokenized_corpus = [self._tokenize(doc) for doc in text_corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        print(f"HybridSearch initialized with {len(text_corpus)} documents")

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into lowercase words."""
        if not text:
            return []
        return re.findall(self.tokenizer_pattern, text.lower())

    def search(
        self,
        query_embedding: np.ndarray,
        query_text: str,
        k: int = 10,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search using RRF.
        
        Args:
            query_embedding: Vector embedding of the query (audio or text)
            query_text: Text description for BM25 matching
            k: Number of results to return
            vector_weight: Weight for vector search scores (not used in RRF, kept for compatibility)
            bm25_weight: Weight for BM25 scores (not used in RRF, kept for compatibility)
        
        Returns:
            List of results with fused scores, each containing:
                - index: document index
                - score: RRF score
                - metadata: associated metadata (if available)
                - bm25_score: original BM25 score
                - vector_rank: rank from vector search
                - bm25_rank: rank from BM25 search
        """
        # 1. Vector search (get more candidates for fusion)
        vector_results, _ = self.vector_index.search(query_embedding, k=2*k)

        # 2. BM25 search
        tokenized_query = self._tokenize(query_text)
        if tokenized_query:
            bm25_scores = self.bm25.get_scores(tokenized_query)
            # Get top 2*k BM25 indices
            bm25_top_indices = np.argsort(bm25_scores)[::-1][:2*k]
        else:
            bm25_scores = np.zeros(len(self.text_corpus))
            bm25_top_indices = []

        # 3. RRF fusion
        # Build rank maps: document index -> rank (0-based)
        vec_rank_map = {}
        if vector_results and len(vector_results) > 0:
            for rank, r in enumerate(vector_results[0]):
                vec_rank_map[r['index']] = rank

        bm25_rank_map = {idx: rank for rank, idx in enumerate(bm25_top_indices)}

        # Collect all unique document indices from both sets
        all_doc_indices = set(vec_rank_map.keys()) | set(bm25_rank_map.keys())

        # Compute RRF scores
        rrf_scores = {}
        for doc_idx in all_doc_indices:
            rrf = 0.0
            if doc_idx in vec_rank_map:
                rrf += 1.0 / (self.k_rrf + vec_rank_map[doc_idx] + 1)
            if doc_idx in bm25_rank_map:
                rrf += 1.0 / (self.k_rrf + bm25_rank_map[doc_idx] + 1)
            rrf_scores[doc_idx] = rrf

        # Sort by RRF score descending
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # Build final results
        final_results = []
        for doc_idx, score in sorted_docs[:k]:
            # Retrieve metadata from vector index if available
            metadata = None
            if hasattr(self.vector_index, 'metadata') and self.vector_index.metadata:
                if doc_idx < len(self.vector_index.metadata):
                    metadata = self.vector_index.metadata[doc_idx]
            
            final_results.append({
                'index': doc_idx,
                'score': score,
                'metadata': metadata,
                'bm25_score': float(bm25_scores[doc_idx]) if doc_idx < len(bm25_scores) else 0.0,
                'vector_rank': vec_rank_map.get(doc_idx, None),
                'bm25_rank': bm25_rank_map.get(doc_idx, None),
            })

        return final_results

    def search_batch(
        self,
        query_embeddings: np.ndarray,
        query_texts: List[str],
        k: int = 10
    ) -> List[List[Dict[str, Any]]]:
        """
        Perform hybrid search for multiple queries in batch.
        
        Args:
            query_embeddings: Array of query embeddings (shape: n_queries x dim)
            query_texts: List of text descriptions for each query
            k: Number of results per query
        
        Returns:
            List of results for each query
        """
        results = []
        for emb, text in zip(query_embeddings, query_texts):
            results.append(self.search(emb, text, k))
        return results

    def add_documents(self, new_texts: List[str]):
        """
        Add new documents to the BM25 index.
        
        Args:
            new_texts: List of text strings to add
        """
        self.text_corpus.extend(new_texts)
        tokenized_new = [self._tokenize(doc) for doc in new_texts]
        # Rebuild BM25 index with all documents
        tokenized_corpus = [self._tokenize(doc) for doc in self.text_corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"Added {len(new_texts)} documents. Total: {len(self.text_corpus)}")
