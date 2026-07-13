import pytest
import numpy as np
from search.faiss_index import FAISSSearch
from search.milvus_index import MilvusSearch
from evaluation.metrics import SearchMetrics

class TestSearch:
    """Unit tests for search functionality"""
    
    @pytest.fixture
    def fake_embeddings(self):
        """Generate fake embeddings for testing"""
        np.random.seed(42)
        return np.random.randn(100, 64).astype(np.float32)
    
    @pytest.fixture
    def fake_metadata(self):
        """Generate fake metadata"""
        return [
            {'id': i, 'track_id': i, 'genre': 'test'}
            for i in range(100)
        ]
    
    def test_faiss_index(self, fake_embeddings, fake_metadata):
        """Test FAISS index operations"""
        index = FAISSSearch(
            dimension=64,
            index_type='HNSW',
            M=16,
            ef_construction=200
        )
        
        # Add vectors
        ids = [m['id'] for m in fake_metadata]
        index.add(fake_embeddings, ids, fake_metadata)
        
        # Test search
        query = fake_embeddings[0]
        results, _ = index.search(query, k=10)
        
        # Should find itself as top result
        assert results[0][0]['id'] == 0
        assert len(results[0]) == 10
        
        # Test save/load
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            index.save(tmpdir)
            new_index = FAISSSearch(dimension=64)
            new_index.load(tmpdir)
            
            # Test search on loaded index
            results2, _ = new_index.search(query, k=10)
            assert results2[0][0]['id'] == 0
    
    def test_search_metrics(self):
        """Test search quality metrics"""
        metrics = SearchMetrics()
        
        ground_truth = [
            [1, 2, 3, 4],
            [5, 6, 7, 8]
        ]
        predictions = [
            [1, 2, 5, 6],
            [5, 6, 1, 2]
        ]
        
        recall = metrics.recall_at_k(ground_truth, predictions, k=2)
        assert recall == 1.0
        
        precision = metrics.precision_at_k(ground_truth, predictions, k=2)
        assert precision == 1.0

if __name__ == "__main__":
    pytest.main([__file__])
