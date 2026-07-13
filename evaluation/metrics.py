import numpy as np
from sklearn.metrics import average_precision_score
from collections import defaultdict
import time

class SearchMetrics:
    """Comprehensive search evaluation metrics"""
    
    @staticmethod
    def recall_at_k(ground_truth, predictions, k):
        """Compute Recall@k"""
        total = 0
        hits = 0
        
        for gt, pred in zip(ground_truth, predictions):
            # gt: list of relevant indices, pred: list of predicted indices
            relevant_set = set(gt[:k])
            predicted_set = set(pred[:k])
            
            hits += len(relevant_set.intersection(predicted_set))
            total += min(len(gt), k)
        
        return hits / total if total > 0 else 0.0
    
    @staticmethod
    def precision_at_k(ground_truth, predictions, k):
        """Compute Precision@k"""
        total = 0
        
        for gt, pred in zip(ground_truth, predictions):
            relevant_set = set(gt[:k])
            predicted_set = set(pred[:k])
            
            total += len(relevant_set.intersection(predicted_set)) / k
        
        return total / len(ground_truth) if ground_truth else 0.0
    
    @staticmethod
    def mean_average_precision(ground_truth, predictions, k=None):
        """Compute mean Average Precision"""
        aps = []
        
        for gt, pred in zip(ground_truth, predictions):
            # Create relevance binary vector
            relevant_indices = set(gt)
            relevance = [1 if p in relevant_indices else 0 for p in pred[:k]]
            
            if sum(relevance) == 0:
                continue
            
            ap = average_precision_score(relevance, [1]*len(relevance))
            aps.append(ap)
        
        return np.mean(aps) if aps else 0.0
    
    @staticmethod
    def ndcg_at_k(ground_truth, predictions, k):
        """Compute NDCG@k (Normalized Discounted Cumulative Gain)"""
        def dcg(relevance_scores):
            return sum(
                rel / np.log2(idx + 2) 
                for idx, rel in enumerate(relevance_scores[:k])
            )
        
        ndcg_scores = []
        
        for gt, pred in zip(ground_truth, predictions):
            relevant_indices = set(gt)
            
            # Compute DCG
            relevance = [1 if p in relevant_indices else 0 for p in pred[:k]]
            dcg_score = dcg(relevance)
            
            # Compute IDCG (ideal DCG)
            ideal_relevance = sorted(relevance, reverse=True)
            idcg_score = dcg(ideal_relevance)
            
            if idcg_score > 0:
                ndcg_scores.append(dcg_score / idcg_score)
            else:
                ndcg_scores.append(0.0)
        
        return np.mean(ndcg_scores) if ndcg_scores else 0.0
    
    @staticmethod
    def queries_per_second(total_queries, total_time):
        """Compute QPS"""
        return total_queries / total_time if total_time > 0 else 0.0
    
    @staticmethod
    def memory_usage(index):
        """Estimate memory usage of index"""
        if hasattr(index, 'index'):
            # FAISS
            if hasattr(index.index, 'ntotal'):
                # Approximate memory usage
                vector_memory = index.index.ntotal * index.dimension * 4  # 4 bytes per float
                return vector_memory
        elif hasattr(index, 'collection'):
            # Milvus
            stats = index.get_stats()
            # Rough estimate
            vector_memory = stats['num_entities'] * index.dimension * 4
            return vector_memory
        
        return 0

class SearchEvaluator:
    """Complete evaluation pipeline"""
    
    def __init__(self, search_index, ground_truth):
        self.search_index = search_index
        self.ground_truth = ground_truth
        self.metrics = SearchMetrics()
    
    def evaluate(self, queries, query_metadata, k_values=[1, 5, 10, 50]):
        """Run complete evaluation"""
        results = {
            'recall': {},
            'precision': {},
            'ndcg': {},
            'map': {},
            'qps': 0.0,
            'memory': 0.0,
            'search_time': 0.0
        }
        
        all_predictions = []
        all_search_times = []
        
        # Perform search for each query
        for i, query in enumerate(queries):
            # Get ground truth for this query
            gt = self.ground_truth.get(query_metadata[i], [])
            
            # Perform search
            predictions, search_time = self.search_index.search(query, k=max(k_values))
            all_predictions.append([p['index'] for p in predictions[0]])
            all_search_times.append(search_time)
        
        # Compute metrics for each k
        for k in k_values:
            results['recall'][k] = self.metrics.recall_at_k(
                self.ground_truth.values(), all_predictions, k
            )
            results['precision'][k] = self.metrics.precision_at_k(
                self.ground_truth.values(), all_predictions, k
            )
            results['ndcg'][k] = self.metrics.ndcg_at_k(
                self.ground_truth.values(), all_predictions, k
            )
        
        # Compute MAP
        results['map'] = self.metrics.mean_average_precision(
            self.ground_truth.values(), all_predictions, k=max(k_values)
        )
        
        # Compute QPS
        total_time = sum(all_search_times)
        results['qps'] = self.metrics.queries_per_second(len(queries), total_time)
        
        # Compute memory usage
        results['memory'] = self.metrics.memory_usage(self.search_index)
        
        return results
