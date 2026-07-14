"""
Full evaluator that runs multiple iterations, logs results, and produces comparison tables.
"""

import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import logging

from .metrics import SearchMetrics, SearchEvaluator

logger = logging.getLogger(__name__)


class FullEvaluator:
    """
    Orchestrates multiple experimental iterations and stores results.
    """

    def __init__(self, results_dir: str = "experiments/results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = SearchMetrics()
        self.all_results = {}

    def evaluate_iteration(
        self,
        name: str,
        search_index,
        queries: np.ndarray,
        query_metadata: List[str],
        ground_truth: Dict[str, List[int]],
        k_values: List[int] = [1, 5, 10, 50]
    ) -> Dict[str, Any]:
        """
        Evaluate a single search index on given queries and ground truth.
        Returns full metrics.
        """
        evaluator = SearchEvaluator(search_index, ground_truth)
        results = evaluator.evaluate(queries, query_metadata, k_values)

        # Add extra stats from the index
        if hasattr(search_index, 'get_stats'):
            stats = search_index.get_stats()
            results['index_stats'] = stats

        results['iteration'] = name
        results['timestamp'] = datetime.now().isoformat()

        self.all_results[name] = results
        self._save_iteration(name, results)

        return results

    def _save_iteration(self, name: str, results: Dict):
        """Save individual iteration results to JSON."""
        iter_dir = self.results_dir / name
        iter_dir.mkdir(parents=True, exist_ok=True)
        with open(iter_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)

    def generate_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Create a summary table comparing all iterations on key metrics.
        """
        summary = {}
        for name, res in self.all_results.items():
            summary[name] = {
                'recall@10': res.get('recall', {}).get(10, 0.0),
                'precision@10': res.get('precision', {}).get(10, 0.0),
                'ndcg@10': res.get('ndcg', {}).get(10, 0.0),
                'map': res.get('map', 0.0),
                'qps': res.get('qps', 0.0),
                'memory_mb': res.get('memory', 0) / (1024 * 1024),
                'total_vectors': res.get('index_stats', {}).get('num_entities', 0),
            }
        # Save summary
        summary_path = self.results_dir / 'summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        return summary

    def print_summary(self):
        """Pretty-print summary table to console."""
        summary = self.generate_summary()
        print("\n" + "="*80)
        print("AURA EXPERIMENT SUMMARY")
        print("="*80)
        for name, metrics in summary.items():
            print(f"\n{name.upper()}:")
            for key, val in metrics.items():
                print(f"  {key}: {val:.4f}")
        print("="*80)

    def load_results(self, iteration_name: str) -> Dict:
        """Load a specific iteration's results from disk."""
        path = self.results_dir / iteration_name / 'results.json'
        if path.exists():
            with open(path, 'r') as f:
                return json.load(f)
        return {}
