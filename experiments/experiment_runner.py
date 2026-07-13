import os
import json
import time
import pickle
from pathlib import Path
import numpy as np
from datetime import datetime
import logging
from tqdm import tqdm

from data.dataset import FMADataset, GTZANDataset
from models.audio_encoder import MERTEncoder, LightweightAudioEncoder, EmbeddingOptimizer
from search.faiss_index import FAISSSearch
from search.milvus_index import MilvusSearch
from evaluation.metrics import SearchEvaluator, SearchMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExperimentRunner:
    """Execute and log all experimental iterations"""
    
    def __init__(self, config_path='config/config.yaml'):
        import yaml
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.results_dir = Path('experiments/results')
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics = SearchMetrics()
        self.all_results = {}
    
    def run_iteration(self, iteration_name):
        """Run a single experimental iteration"""
        logger.info(f"=== Running Iteration: {iteration_name} ===")
        
        iteration_config = self._get_iteration_config(iteration_name)
        
        # Phase 1: Generate or load embeddings
        embeddings, metadata = self._prepare_data(iteration_config)
        
        # Phase 2: Build search index
        search_index = self._build_index(embeddings, metadata, iteration_config)
        
        # Phase 3: Evaluate
        evaluation_results = self._evaluate(
            search_index, 
            iteration_config,
            n_queries=1000
        )
        
        # Phase 4: Log results
        self.all_results[iteration_name] = {
            'config': iteration_config,
            'evaluation': evaluation_results,
            'timestamp': datetime.now().isoformat()
        }
        
        self._save_results(iteration_name)
        
        return evaluation_results
    
    def _get_iteration_config(self, name):
        """Get configuration for a specific iteration"""
        for iter_config in self.config['iterations']:
            if iter_config['name'] == name:
                return iter_config
        raise ValueError(f"Iteration {name} not found in config")
    
    def _prepare_data(self, config):
        """Prepare embeddings and metadata for the iteration"""
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
            # Use MERT
            encoder = MERTEncoder()
            
            # Load dataset
            dataset = FMADataset(
                self.config['data']['fma_path'],
                chunk_duration=self.config['data']['chunk_duration'],
                sample_rate=self.config['data']['sample_rate'],
                chunk_stride=self.config['data']['chunk_stride']
            )
            
            embeddings = []
            metadata = []
            
            for idx in tqdm(range(len(dataset))):
                item = dataset[idx]
                if item is None:
                    continue
                
                # Encode each chunk
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
            # Use CLAP (simplified)
            # In practice, implement CLAP encoding here
            pass
        
        # Apply PCA if needed
        if config['model'] == 'mert_pca':
            optimizer = EmbeddingOptimizer(
                original_dim=768,
                target_dim=config.get('dim', 128)
            )
            optimizer.fit_pca(embeddings[:10000])  # Fit on subset
            embeddings = optimizer.transform(embeddings)
            embeddings = optimizer.quantize(embeddings)
            logger.info(f"Reduced embeddings to {embeddings.shape[1]} dims")
        
        # Cache embeddings
        with open(cache_path, 'wb') as f:
            pickle.dump({
                'embeddings': embeddings,
                'metadata': metadata
            }, f)
        
        return embeddings, metadata
    
    def _build_index(self, embeddings, metadata, config):
        """Build search index for the iteration"""
        if config['index'] == 'faiss':
            index = FAISSSearch(
                dimension=embeddings.shape[1],
                index_type='HNSW',
                **self.config['search']['faiss']
            )
            
            ids = [m['track_id'] for m in metadata]
            index.add(embeddings, ids, metadata)
            
        elif config['index'] == 'milvus':
            # In practice, start Milvus server first
            index = MilvusSearch(
                collection_name=f"music_{config['name']}",
                dimension=embeddings.shape[1],
                **self.config['search']['milvus']
            )
            
            # Prepare metadata in batch
            batch_size = 1000
            for i in range(0, len(embeddings), batch_size):
                batch_emb = embeddings[i:i+batch_size]
                batch_meta = metadata[i:i+batch_size]
                
                meta_dict = {
                    'track_id': [m['track_id'] for m in batch_meta],
                    'genre': [m['genre'] for m in batch_meta],
                    'title': [m['title'] for m in batch_meta],
                    'artist': [m['artist'] for m in batch_meta]
                }
                
                index.add(batch_emb, meta_dict)
        
        return index
    
    def _evaluate(self, index, config, n_queries=1000):
        """Evaluate the search index"""
        logger.info("Loading GTZAN queries...")
        
        # Load GTZAN dataset
        gtzan = GTZANDataset(self.config['data']['gtzan_path'])
        
        # Select subset of queries
        n_queries = min(n_queries, len(gtzan))
        queries = []
        query_metadata = []
        ground_truth = {}
        
        for i in tqdm(range(n_queries)):
            item = gtzan[i]
            
            # Encode query audio
            encoder = MERTEncoder()
            emb = encoder.encode_audio(item['audio'])
            queries.append(emb[0])
            
            # Store metadata
            query_metadata.append(item['genre'])
            
            # Get ground truth: tracks with same genre
            # This is a simplification; in practice, use actual relevance judgments
            if item['genre'] not in ground_truth:
                ground_truth[item['genre']] = []
            
            # For demo, use genre-based ground truth
            # In real evaluation, use human judgments or cross-validation
            if i < 10:  # Build ground truth for first few queries
                # Search for similar genres
                results, _ = index.search(emb[0], k=10)
                ground_truth[item['genre']].extend([r['id'] for r in results[0][:5]])
        
        # Evaluate
        evaluator = SearchEvaluator(index, ground_truth)
        results = evaluator.evaluate(
            np.array(queries),
            query_metadata,
            k_values=[1, 5, 10, 50]
        )
        
        logger.info(f"Results: {json.dumps(results, indent=2)}")
        return results
    
    def _save_results(self, name):
        """Save results to file"""
        results_path = self.results_dir / name / 'results.json'
        with open(results_path, 'w') as f:
            json.dump(self.all_results[name], f, indent=2)
        
        logger.info(f"Saved results to {results_path}")
    
    def run_all(self):
        """Run all iterations"""
        for iter_config in self.config['iterations']:
            self.run_iteration(iter_config['name'])
        
        self._generate_summary()
    
    def _generate_summary(self):
        """Generate summary comparison of all iterations"""
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
        logger.info("=" * 50)
        logger.info("EXPERIMENT SUMMARY")
        logger.info("=" * 50)
        for name, metrics in summary.items():
            logger.info(f"\n{name.upper()}:")
            for key, value in metrics.items():
                logger.info(f"  {key}: {value:.4f}")
        
        return summary

if __name__ == "__main__":
    runner = ExperimentRunner()
    runner.run_all()
