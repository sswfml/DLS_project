import faiss
import numpy as np
from pathlib import Path
import pickle
from tqdm import tqdm
import time

class FAISSSearch:
    """FAISS-based vector search implementation"""
    
    def __init__(self, dimension, index_type='HNSW', **kwargs):
        self.dimension = dimension
        self.index_type = index_type
        self.kwargs = kwargs
        
        self.index = None
        self.ids = None
        self.metadata = None
        
        self._create_index()
    
    def _create_index(self):
        """Create FAISS index based on type"""
        if self.index_type == 'HNSW':
            # HNSW index
            index = faiss.IndexHNSWFlat(self.dimension, self.kwargs.get('M', 16))
            index.hnsw.efConstruction = self.kwargs.get('ef_construction', 200)
            index.hnsw.efSearch = self.kwargs.get('ef_search', 100)
            self.index = index
            
        elif self.index_type == 'IVF':
            # IVF index (requires training)
            nlist = self.kwargs.get('nlist', 100)
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            
        elif self.index_type == 'IVFPQ':
            # IVF-PQ index (memory efficient)
            nlist = self.kwargs.get('nlist', 100)
            m = self.kwargs.get('m', 8)
            nbits = self.kwargs.get('nbits', 8)
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFPQ(quantizer, self.dimension, nlist, m, nbits)
            
        else:
            # Default: flat index (exact search)
            self.index = faiss.IndexFlatL2(self.dimension)
        
        print(f"Created {self.index_type} index with dimension {self.dimension}")
    
    def train(self, embeddings):
        """Train index if needed (IVF variants)"""
        if hasattr(self.index, 'train') and not self.index.is_trained:
            print("Training index...")
            self.index.train(embeddings.astype(np.float32))
            print("Index trained")
    
    def add(self, embeddings, ids=None, metadata=None):
        """Add vectors to index"""
        embeddings = embeddings.astype(np.float32)
        
        if len(embeddings.shape) == 1:
            embeddings = embeddings.reshape(1, -1)
        
        self.train(embeddings)
        
        # Add vectors
        self.index.add(embeddings)
        
        # Store IDs and metadata
        if ids is not None:
            if self.ids is None:
                self.ids = []
            self.ids.extend(ids)
        
        if metadata is not None:
            if self.metadata is None:
                self.metadata = []
            self.metadata.extend(metadata)
        
        print(f"Added {embeddings.shape[0]} vectors. Total: {self.index.ntotal}")
    
    def search(self, query, k=10, ef_search=None):
        """Search for nearest neighbors"""
        if len(query.shape) == 1:
            query = query.reshape(1, -1)
        
        query = query.astype(np.float32)
        
        # Temporarily adjust efSearch for HNSW if specified
        if self.index_type == 'HNSW' and ef_search is not None:
            original_ef = self.index.hnsw.efSearch
            self.index.hnsw.efSearch = ef_search
        
        # Search
        start_time = time.time()
        distances, indices = self.index.search(query, k)
        search_time = time.time() - start_time
        
        # Restore efSearch
        if self.index_type == 'HNSW' and ef_search is not None:
            self.index.hnsw.efSearch = original_ef
        
        # Map to IDs and metadata
        results = []
        for i in range(len(query)):
            query_results = []
            for j, idx in enumerate(indices[i]):
                if idx == -1:  # FAISS returns -1 for invalid indices
                    continue
                
                result = {
                    'index': int(idx),
                    'distance': float(distances[i][j]),
                    'id': self.ids[idx] if self.ids else idx,
                    'metadata': self.metadata[idx] if self.metadata else None
                }
                query_results.append(result)
            results.append(query_results)
        
        return results, search_time
    
    def save(self, path):
        """Save index to disk"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(path / 'index.faiss'))
        
        # Save IDs and metadata
        with open(path / 'metadata.pkl', 'wb') as f:
            pickle.dump({
                'ids': self.ids,
                'metadata': self.metadata,
                'dimension': self.dimension,
                'index_type': self.index_type
            }, f)
        
        print(f"Saved index to {path}")
    
    def load(self, path):
        """Load index from disk"""
        path = Path(path)
        
        # Load FAISS index
        self.index = faiss.read_index(str(path / 'index.faiss'))
        
        # Load IDs and metadata
        with open(path / 'metadata.pkl', 'rb') as f:
            data = pickle.load(f)
            self.ids = data['ids']
            self.metadata = data['metadata']
            self.dimension = data['dimension']
            self.index_type = data['index_type']
        
        print(f"Loaded index from {path} (ntotal={self.index.ntotal})")
    
    def get_stats(self):
        """Get index statistics"""
        return {
            'total_vectors': self.index.ntotal,
            'dimension': self.dimension,
            'index_type': self.index_type,
            'is_trained': self.index.is_trained
        }
