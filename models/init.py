"""Model definitions and embedding optimization for AURA."""

from .audio_encoder import MERTEncoder, LightweightAudioEncoder
from .embedding_optimizer import EmbeddingOptimizer
from .clap_adapter import CLAPAdapter

__all__ = [
    "MERTEncoder",
    "LightweightAudioEncoder",
    "EmbeddingOptimizer",
    "CLAPAdapter",
]
