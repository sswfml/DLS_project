"""Optimize embeddings via PCA, quantization, and distillation."""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple
import pickle


class EmbeddingOptimizer:
    """
    Reduce dimensionality and quantize embeddings.
    Saves scaling parameters for later dequantization.
    """

    def __init__(self, original_dim: int, target_dim: int):
        self.original_dim = original_dim
        self.target_dim = target_dim
        self.pca: Optional[PCA] = None
        self.scaler: Optional[StandardScaler] = None
        self.quant_params: Optional[dict] = None

    def fit_pca(self, embeddings: np.ndarray, n_components: Optional[int] = None) -> "EmbeddingOptimizer":
        """Fit PCA on a sample of embeddings."""
        n_components = n_components or self.target_dim

        self.scaler = StandardScaler()
        scaled = self.scaler.fit_transform(embeddings)

        self.pca = PCA(n_components=n_components)
        self.pca.fit(scaled)

        explained = self.pca.explained_variance_ratio_.sum()
        print(f"PCA explained variance: {explained:.4f} (target dim: {n_components})")
        return self

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply PCA transformation."""
        if self.pca is None:
            raise ValueError("PCA not fitted. Call fit_pca first.")
        scaled = self.scaler.transform(embeddings)
        reduced = self.pca.transform(scaled)
        return reduced

    def quantize(self, embeddings: np.ndarray, dtype=np.int8) -> np.ndarray:
        """
        Quantize to int8 (or other integer type) using min-max scaling.
        Stores scaling params for dequantization.
        """
        if dtype != np.int8:
            return embeddings.astype(dtype)

        min_vals = embeddings.min(axis=0)
        max_vals = embeddings.max(axis=0)
        ranges = max_vals - min_vals
        ranges[ranges == 0] = 1.0  # avoid division by zero

        # Scale to [-128, 127]
        quantized = ((embeddings - min_vals) / ranges) * 255 - 128
        quantized = np.round(quantized).astype(np.int8)

        self.quant_params = {"min": min_vals, "range": ranges}
        return quantized

    def dequantize(self, quantized: np.ndarray) -> np.ndarray:
        """Recover float embeddings from int8 quantized representation."""
        if self.quant_params is None:
            raise ValueError("Quantization parameters not stored.")
        return ((quantized.astype(np.float32) + 128) / 255) * self.quant_params["range"] + self.quant_params["min"]

    def save(self, path: str):
        """Save optimizer state (PCA, scaler, quant params)."""
        with open(path, 'wb') as f:
            pickle.dump({
                'pca': self.pca,
                'scaler': self.scaler,
                'quant_params': self.quant_params,
                'original_dim': self.original_dim,
                'target_dim': self.target_dim,
            }, f)

    def load(self, path: str):
        """Load optimizer state."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.pca = data['pca']
        self.scaler = data['scaler']
        self.quant_params = data['quant_params']
        self.original_dim = data['original_dim']
        self.target_dim = data['target_dim']
        return self
