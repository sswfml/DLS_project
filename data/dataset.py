import os
import numpy as np
import librosa
import torch
import torchaudio
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class FMADataset(Dataset):
    """FMA Dataset with chunking for embedding generation"""
    
    def __init__(self, fma_path, chunk_duration=10, sample_rate=16000, 
                 chunk_stride=5, augment=True):
        self.fma_path = Path(fma_path)
        self.chunk_duration = chunk_duration
        self.sample_rate = sample_rate
        self.chunk_stride = chunk_stride
        self.augment = augment
        
        # Load track metadata
        self.tracks = pd.read_csv(self.fma_path / 'tracks.csv')
        self.tracks = self.tracks[self.tracks['set', 'subset'] == 'medium']
        
        # Filter for MP3 files
        self.tracks['mp3_exists'] = self.tracks.apply(
            lambda row: (self.fma_path / 'fma_medium' / f"{row['track', 'id']:06d}.mp3").exists(),
            axis=1
        )
        self.tracks = self.tracks[self.tracks['mp3_exists']]
        
        self.track_ids = self.tracks.index.get_level_values('track_id').tolist()
        print(f"Found {len(self.track_ids)} tracks with MP3 files")
    
    def _load_audio(self, track_id):
        """Load and preprocess audio from MP3"""
        track_path = self.fma_path / 'fma_medium' / f"{track_id:06d}.mp3"
        
        try:
            audio, sr = librosa.load(track_path, sr=self.sample_rate, mono=True)
            return audio
        except Exception as e:
            print(f"Error loading track {track_id}: {e}")
            return None
    
    def _chunk_audio(self, audio):
        """Split audio into overlapping chunks"""
        chunk_len = self.chunk_duration * self.sample_rate
        stride_len = self.chunk_stride * self.sample_rate
        
        if len(audio) < chunk_len:
            # Pad if too short
            audio = np.pad(audio, (0, chunk_len - len(audio)))
        
        chunks = []
        for start in range(0, len(audio) - chunk_len + 1, stride_len):
            chunk = audio[start:start + chunk_len]
            chunks.append(chunk)
        
        return chunks
    
    def _augment_audio(self, audio):
        """Apply data augmentation"""
        if not self.augment:
            return [audio]
        
        augmented = [audio]  # original
        
        # Pitch shift (semitone steps)
        for semitone in [-2, 2]:
            augmented.append(librosa.effects.pitch_shift(audio, sr=self.sample_rate, n_steps=semitone))
        
        # Time stretch
        for rate in [0.9, 1.1]:
            augmented.append(librosa.effects.time_stretch(audio, rate=rate))
        
        # Add noise
        noise = np.random.randn(len(audio)) * 0.001
        augmented.append(audio + noise)
        
        return augmented
    
    def __len__(self):
        return len(self.track_ids)
    
    def __getitem__(self, idx):
        track_id = self.track_ids[idx]
        audio = self._load_audio(track_id)
        
        if audio is None:
            return None
        
        # Get track metadata
        metadata = self.tracks.loc[track_id]
        
        # Chunk the audio
        chunks = self._chunk_audio(audio)
        
        # Augment each chunk
        all_chunks = []
        for chunk in chunks:
            all_chunks.extend(self._augment_audio(chunk))
        
        return {
            'track_id': track_id,
            'chunks': all_chunks,
            'metadata': {
                'genre_top': metadata['track', 'genre_top'],
                'duration': metadata['track', 'duration'],
                'title': metadata['track', 'title'],
                'artist': metadata['artist', 'name']
            }
        }

class GTZANDataset(Dataset):
    """GTZAN dataset for validation queries"""
    
    def __init__(self, gtzan_path, sample_rate=16000):
        self.gtzan_path = Path(gtzan_path)
        self.sample_rate = sample_rate
        
        # List all audio files
        self.audio_files = sorted(self.gtzan_path.glob('genres/*/*.wav'))
        print(f"Found {len(self.audio_files)} GTZAN tracks")
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        
        # Ensure 10 seconds
        target_len = 10 * self.sample_rate
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))
        else:
            audio = audio[:target_len]
        
        # Get genre label from path
        genre = audio_path.parent.name
        
        return {
            'audio': audio,
            'genre': genre,
            'path': str(audio_path)
        }
