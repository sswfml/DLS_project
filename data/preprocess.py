"""Audio preprocessing utilities: chunking, augmentation, normalization."""

import numpy as np
import librosa
from typing import List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class AudioPreprocessor:
    """
    Handles loading, resampling, and basic preprocessing of audio files.
    """

    def __init__(self, target_sr: int = 16000, mono: bool = True):
        self.target_sr = target_sr
        self.mono = mono

    def load(self, file_path: str, duration: Optional[float] = None) -> np.ndarray:
        """
        Load audio file, resample to target_sr, convert to mono.
        If duration is given, load only that many seconds.
        """
        audio, sr = librosa.load(
            file_path,
            sr=self.target_sr,
            mono=self.mono,
            duration=duration
        )
        return audio

    def normalize(self, audio: np.ndarray, method: str = "peak") -> np.ndarray:
        """Normalize audio: 'peak' or 'rms'."""
        if method == "peak":
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                return audio / max_val
        elif method == "rms":
            rms = np.sqrt(np.mean(audio ** 2))
            if rms > 0:
                return audio / (rms + 1e-8)
        return audio

    def pad_or_trim(self, audio: np.ndarray, target_length: int) -> np.ndarray:
        """Pad or trim audio to exact target_length samples."""
        if len(audio) < target_length:
            pad_width = target_length - len(audio)
            audio = np.pad(audio, (0, pad_width), mode='constant')
        else:
            audio = audio[:target_length]
        return audio


class ChunkGenerator:
    """
    Generate overlapping chunks from an audio signal.
    Supports optional augmentation.
    """

    def __init__(
        self,
        chunk_duration: float = 10.0,
        sample_rate: int = 16000,
        stride_duration: float = 5.0,
        augment: bool = True
    ):
        self.chunk_len = int(chunk_duration * sample_rate)
        self.stride_len = int(stride_duration * sample_rate)
        self.sample_rate = sample_rate
        self.augment = augment

    def generate_chunks(self, audio: np.ndarray) -> List[np.ndarray]:
        """Split audio into overlapping chunks."""
        chunks = []
        if len(audio) < self.chunk_len:
            # Pad to minimum length
            audio = np.pad(audio, (0, self.chunk_len - len(audio)))
        for start in range(0, len(audio) - self.chunk_len + 1, self.stride_len):
            chunk = audio[start:start + self.chunk_len]
            chunks.append(chunk)
        return chunks

    def augment_chunk(self, chunk: np.ndarray) -> List[np.ndarray]:
        """
        Apply pitch shift, time stretch, and noise augmentation.
        Returns list of augmented versions (including original).
        """
        if not self.augment:
            return [chunk]

        augmented = [chunk]

        # Pitch shift (±2 semitones)
        for semitone in [-2, 2]:
            shifted = librosa.effects.pitch_shift(chunk, sr=self.sample_rate, n_steps=semitone)
            augmented.append(shifted)

        # Time stretch (0.9x and 1.1x)
        for rate in [0.9, 1.1]:
            stretched = librosa.effects.time_stretch(chunk, rate=rate)
            # Resize to original length
            stretched = librosa.util.fix_length(stretched, size=len(chunk))
            augmented.append(stretched)

        # Add Gaussian noise (small amplitude)
        noise = np.random.randn(len(chunk)) * 0.001 * np.max(np.abs(chunk))
        augmented.append(chunk + noise)

        return augmented

    def process(self, audio: np.ndarray) -> List[np.ndarray]:
        """Full pipeline: chunk + augment each chunk."""
        chunks = self.generate_chunks(audio)
        all_chunks = []
        for c in chunks:
            all_chunks.extend(self.augment_chunk(c))
        return all_chunks
