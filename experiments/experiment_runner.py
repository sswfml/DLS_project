"""
AURA Experiment Runner - orchestrates all iterations.
"""

import os
import json
import time
import pickle
from pathlib import Path
import numpy as np
from datetime import datetime
import logging
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from data.dataset import FMADataset, GTZANDataset
from models.audio_encoder import MERTEncoder
from models.embedding_optimizer import EmbeddingOptimizer
from models.clap_adapter import CLAPAdapter
from search.faiss_index import FAISSSearch
from search.milvus_index import MilvusSearch
from search.hybrid_search import HybridSearch
from evaluation.metrics import SearchMetrics, SearchEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AURAExperimentRunner:
    """Execute and log all experimental iterations for AURA."""

    def __init__(self, config_path='config/config.yaml'):
        import yaml
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.results_dir = Path('experiments/results')
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.metrics = SearchMetrics()
        self.all_results = {}

    def run_iteration(self, iteration_name):
        """Run a single experimental iteration."""
        logger.info(f"🎵 Running AURA Iteration: {iteration_name}")
        logger.info("=" * 60)

        iteration_config = self._get_iteration_config(iteration_name)

        # Phase 1: Generate or load embeddings
        embeddings, metadata = self._prepare_data(iteration_config)

        # Phase 2: Build search index
        search_index = self._build_index(embeddings, metadata, iteration_config)

        # Phase 3: Evaluate
        evaluation_results = self._evaluate(
            search_index,
            iteration_config,
            n_queries=self.config['evaluation']['n_queries']
        )

        # Phase 4: Log results
        self.all_results[iteration_name] = {
            'config': iteration_config,
            'evaluation': evaluation_results,
            'timestamp': datetime.now().isoformat()
        }

        self._save_results(iteration_name)
        logger.info(f"✅ Iteration {iteration_name} complete")

        return evaluation_results

    def _get_iteration_config(self, name):
        """Get configuration for a specific iteration."""
        for iter_config in self.config['iterations']:
            if iter_config['name'] == name:
                return iter_config
        raise ValueError(f"Iteration {name} not found in config")

    def _prepare_data(self, config):
        """Prepare embeddings and metadata for the iteration."""
        cache_path = self.results_dir / config['name'] / 'embeddings.pkl'
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Check cache
        if cache_path.exists():
            logger.info(f"Loading cached embeddings from {cache_path}")
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
                return data['embeddings'], data['metadata']

        logger.info("Generating embeddings...")

        if config['model'] == 'mert' or config['model'] == 'mert_pca':
            encoder = MERTEncoder()

            dataset = FMADataset(
                self.config['data']['fma_path'],
                chunk_duration=self.config['data']['chunk_duration'],
                sample_rate=self.config['data']['sample_rate'],
                chunk_stride=self.config['data']['chunk_stride']
            )

            embeddings = []
            metadata = []

            for idx in tqdm(range(len(dataset)), desc="Processing tracks"):
                item = dataset[idx]
                if item is None:
                    continue

                for chunk in item['chunks']:
                    emb = encoder.encode_audio(chunk)
                    embeddings.append(emb[0])
                    metadata.append({
                        'track_id': item['track_id'],
                        'genre': item['metadata']['genre_top'],
                        'title': item['metadata']['title'],
                        'artist': item['metadata']['artist']
                    })

            embeddings = np.array(embeddings)

        elif config['model'] == 'clap':
            # CLAP encoding
            clap = CLAPAdapter()
            dataset = FMADataset(
                self.config['data']['fma_path'],
                chunk_duration=self.config['data']['chunk_duration'],
                sample_rate=48000,  # CLAP default
            )
            embeddings = []
            metadata = []
            for idx in tqdm(range(len(dataset)), desc="CLAP encoding"):
                item = dataset[idx]
                if item is None:
                    continue
                for chunk in item['chunks']:
                    emb = clap.encode_audio(chunk)
                    embeddings.append(emb[0])
                    metadata.append({
                        'track_id': item['track_id'],
                        'genre': item['metadata']['genre_top'],
                        'title': item['metadata']['title'],
                        'artist': item['metadata']['artist']
                    })
            embeddings = np.array(embeddings)

        # Apply PCA if needed
        if config['model'] == 'mert_pca':
            optimizer = EmbeddingOptimizer(
                original_dim=768,
                target_dim=config.get('dim', 128)
            )
            logger.info("Fitting PCA on embeddings subset...")
            optimizer.fit_pca(embeddings[:10000])
            embeddings = optimizer.transform(embeddings)
            embeddings = optimizer.quantize(embeddings)
            logger.info(f"Reduced embeddings to {embeddings.shape[1]} dims, quantized to int8")

        # Cache embeddings
        with open(cache_path, 'wb') as f:
            pickle.dump({
                'embeddings': embeddings,
                'metadata': metadata
            }, f)

        return embeddings, metadata

    def _build_index(self, embeddings, metadata, config):
        """Build search index for the iteration."""
        logger.info(f"Building {config['index']} index...")

        if config['index'] == 'faiss':
            index = FAISSSearch(
                dimension=embeddings.shape[1],
                index_type='HNSW',
                **self.config['search']['faiss']
            )

            ids = [m['track_id'] for m in metadata]
            index.add(embeddings, ids, metadata)

        elif config['index'] == 'milvus':
            # Try to connect to Milvus
            try:
                index = MilvusSearch(
                    collection_name=f"aura_music_{config['name']}",
                    dimension=embeddings.shape[1],
                    **self.config['search']['milvus']
                )

                # Prepare metadata in batch
                batch_size = 1000
                for i in tqdm(range(0, len(embeddings), batch_size), desc="Adding to Milvus"):
                    batch_emb = embeddings[i:i+batch_size]
                    batch_meta = metadata[i:i+batch_size]

                    meta_dict = {
                        'track_id': [m['track_id'] for m in batch_meta],
                        'genre': [m['genre'] for m in batch_meta],
                        'title': [m['title'] for m in batch_meta],
                        'artist': [m['artist'] for m in batch_meta]
                    }

                    index.add(batch_emb, meta_dict)
            except Exception as e:
                logger.warning(f"Milvus not available: {e}")
                logger.warning("Falling back to FAISS...")
                index = FAISSSearch(
                    dimension=embeddings.shape[1],
                    index_type='IVFPQ',
                    **self.config['search']['faiss']
                )
                ids = [m['track_id'] for m in metadata]
                index.add(embeddings, ids, metadata)

        return index

    def _evaluate(self, index, config, n_queries=1000):
        """Evaluate the search index."""
        logger.info("Loading GTZAN queries...")

        gtzan = GTZANDataset(self.config['data']['gtzan_path'])

        n_queries = min(n_queries, len(gtzan))
        queries = []
        query_metadata = []
        ground_truth = {}

        # Create ground truth using genre labels
        genre_to_tracks = {}
        for i in range(len(gtzan)):
            item = gtzan[i]
            genre = item['genre']
            if genre not in genre_to_tracks:
                genre_to_tracks[genre] = []
            genre_to_tracks[genre].append(i)

        # Generate embeddings for queries
        encoder = MERTEncoder()
        for i in tqdm(range(n_queries), desc="Encoding queries"):
            item = gtzan[i]
            emb = encoder.encode_audio(item['audio'])
            queries.append(emb[0])
            query_metadata.append(item['genre'])

            # Ground truth: other tracks with same genre
            if item['genre'] in ground_truth:
                # Use first 10 tracks of same genre as relevant
                relevant = genre_to_tracks.get(item['genre'], [])
                ground_truth[i] = relevant[:10]

        # Evaluate
        evaluator = SearchEvaluator(index, ground_truth)
        results = evaluator.evaluate(
            np.array(queries),
            query_metadata,
            k_values=self.config['evaluation']['recall_k']
        )

        logger.info(f"Results: recall@10={results['recall'].get(10, 0):.4f}, "
                    f"qps={results['qps']:.2f}, memory={results['memory']/(1024*1024):.2f}MB")

        return results

    def _save_results(self, name):
        """Save results to file."""
        results_path = self.results_dir / name / 'results.json'
        with open(results_path, 'w') as f:
            json.dump(self.all_results[name], f, indent=2, default=str)

        logger.info(f"Saved results to {results_path}")

    def run_all(self):
        """Run all iterations."""
        for iter_config in self.config['iterations']:
            self.run_iteration(iter_config['name'])

        self._generate_summary()

    def _generate_summary(self):
        """Generate summary comparison of all iterations."""
        summary = {}

        for name, results in self.all_results.items():
            eval_results = results['evaluation']
            summary[name] = {
                'recall@10': eval_results['recall'].get(10, 0),
                'precision@10': eval_results['precision'].get(10, 0),
                'ndcg@10': eval_results['ndcg'].get(10, 0),
                'map': eval_results['map'],
                'qps': eval_results['qps'],
                'memory_mb': eval_results['memory'] / (1024 * 1024)
            }

        # Save summary
        summary_path = self.results_dir / 'summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        # Print summary
        logger.info("=" * 60)
        logger.info("AURA EXPERIMENT SUMMARY")
        logger.info("=" * 60)
        for name, metrics in summary.items():
            logger.info(f"\n{name.upper()}:")
            for key, value in metrics.items():
                logger.info(f"  {key}: {value:.4f}")
        logger.info("=" * 60)

        return summary


if __name__ == "__main__":
    runner = AURAExperimentRunner()
    runner.run_all()
