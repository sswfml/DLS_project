"""CLAP (Contrastive Language-Audio Pretraining) adapter for text-to-audio search."""

import torch
import numpy as np
from transformers import ClapModel, ClapProcessor
from typing import Union, List
import warnings
warnings.filterwarnings('ignore')


class CLAPAdapter:
    """
    Wrapper for CLAP model to generate text and audio embeddings
    for multimodal search.
    """

    def __init__(self, model_name: str = "laion/clap", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading CLAP model on {self.device}...")
        self.model = ClapModel.from_pretrained(model_name).to(self.device)
        self.processor = ClapProcessor.from_pretrained(model_name)
        self.model.eval()
        self.embedding_dim = 512  # CLAP's audio/text projection size

    @torch.no_grad()
    def encode_text(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate text embeddings for one or more text prompts.
        """
        if isinstance(texts, str):
            texts = [texts]

        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        outputs = self.model.get_text_features(**inputs)
        return outputs.cpu().numpy()

    @torch.no_grad()
    def encode_audio(self, audio_arrays: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """
        Generate audio embeddings for a batch of audio arrays.
        Each array should be 1D and sampled at 48kHz (CLAP default) or resampled.
        """
        if isinstance(audio_arrays, np.ndarray) and audio_arrays.ndim == 1:
            audio_arrays = [audio_arrays]
        elif isinstance(audio_arrays, list):
            pass
        else:
            raise ValueError("audio_arrays must be a list of 1D arrays or a single 1D array.")

        # CLAP expects sampling rate 48000, we'll resample if needed.
        # For simplicity, assume already at 48kHz.
        inputs = self.processor(
            audios=audio_arrays,
            return_tensors="pt",
            sampling_rate=48000,
            padding=True,
        ).to(self.device)

        outputs = self.model.get_audio_features(**inputs)
        return outputs.cpu().numpy()

    @torch.no_grad()
    def similarity(self, text_emb: np.ndarray, audio_emb: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between text and audio embeddings.
        Returns matrix of shape (len(text), len(audio)).
        """
        # Normalize
        text_emb = text_emb / np.linalg.norm(text_emb, axis=1, keepdims=True)
        audio_emb = audio_emb / np.linalg.norm(audio_emb, axis=1, keepdims=True)
        return np.dot(text_emb, audio_emb.T)
