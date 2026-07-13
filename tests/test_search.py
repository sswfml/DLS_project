"""
Unit tests for AURA search functionality.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from search.faiss_index import FAISSSearch
from evaluation.metrics import SearchMetrics
from search.hybrid_search import HybridSearch


class TestAURASearch:
    """Unit tests for AURA search components."""

    @pytest.fixture
    def fake_embeddings(self):
        """Generate fake embeddings for testing."""
        np.random.seed(42)
        return np.random.randn(100, 64).astype(np.float32)

    @pytest.fixture
    def fake_metadata(self):
        """Generate fake metadata."""
        return [
            {
                'id': i,
                'track_id': i,
                'genre': ['pop', 'rock', 'jazz'][i % 3],
                'title': f'Track_{i}',
                'artist': f'Artist_{i % 10}'
            }
            for i in range(100)
        ]

    @pytest.fixture
    def fake_text_corpus(self):
        """Generate fake text corpus for hybrid search."""
        return [
            "upbeat pop song",
            "rock ballad with guitar",
            "jazz improvisation",
            "electronic dance music",
            "classical symphony"
        ] * 20

    def test_faiss_index_operations(self, fake_embeddings, fake_metadata):
        """Test FAISS index operations."""
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
        results, search_time = index.search(query, k=10)

        # Should find itself as top result
        assert results[0][0]['id'] == 0
        assert len(results[0]) == 10
        assert search_time > 0

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
        """Test search quality metrics."""
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

    def test_hybrid_search(self, fake_embeddings, fake_metadata, fake_text_corpus):
        """Test hybrid search functionality."""
        # Build a simple FAISS index
        index = FAISSSearch(dimension=64, index_type='Flat')
        ids = [m['id'] for m in fake_metadata]
        index.add(fake_embeddings, ids, fake_metadata)

        # Create hybrid search
        hybrid = HybridSearch(index, fake_text_corpus)

        # Test search
        query_embedding = fake_embeddings[0]
        results = hybrid.search(
            query_embedding,
            query_text="pop song",
            k=5
        )

        assert len(results) == 5
        assert 'index' in results[0]
        assert 'score' in results[0]
        assert 'metadata' in results[0]

    def test_memory_estimation(self, fake_embeddings):
        """Test memory estimation."""
        index = FAISSSearch(dimension=64, index_type='Flat')
        index.add(fake_embeddings)

        from evaluation.metrics import SearchMetrics
        memory = SearchMetrics.memory_usage(index)
        assert memory > 0
        assert memory < fake_embeddings.nbytes * 2  # Rough estimate


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
