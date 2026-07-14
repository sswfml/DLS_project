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

        # Check if columns are MultiIndex (real FMA) or simple (fake data)
        if isinstance(self.tracks.columns, pd.MultiIndex):
            # Real FMA data
            self.tracks = self.tracks[self.tracks['set', 'subset'] == 'medium']
            self.tracks['mp3_exists'] = self.tracks.apply(
                lambda row: (self.fma_path / 'fma_medium' / f"{row['track', 'id']:06d}.mp3").exists(),
                axis=1
            )
            self.tracks = self.tracks[self.tracks['mp3_exists']]
            self.track_ids = self.tracks.index.get_level_values('track_id').tolist()
        else:
            # Simple/fake data – handle gracefully
            # Determine the ID column
            id_col = None
            for col in ['track_id', 'id', 'track']:
                if col in self.tracks.columns:
                    id_col = col
                    break
            if id_col is None:
                id_col = self.tracks.columns[0]
            
            # For fake data, we assume all files exist (or skip the check)
            # So we set mp3_exists to True for all rows
            self.tracks['mp3_exists'] = True
            
            # If you actually want to check files (optional), convert to int safely:
            # def file_exists(row):
            #     try:
            #         return (self.fma_path / 'fma_medium' / f"{int(row[id_col]):06d}.mp3").exists()
            #     except:
            #         return False
            # self.tracks['mp3_exists'] = self.tracks.apply(file_exists, axis=1)
            
            # Filter (but all are True anyway)
            self.tracks = self.tracks[self.tracks['mp3_exists']]
            self.track_ids = self.tracks[id_col].tolist()

        print(f"Found {len(self.track_ids)} tracks with MP3 files")
    
    def _load_audio(self, track_id):
        """Load and preprocess audio from MP3"""
        try:
            # Try to convert track_id to int for formatting
            track_id_int = int(track_id)
            track_path = self.fma_path / 'fma_medium' / f"{track_id_int:06d}.mp3"
            audio, sr = librosa.load(track_path, sr=self.sample_rate, mono=True)
            return audio
        except Exception as e:
            # For fake data, return random noise
            print(f"Warning: Could not load track {track_id}, using random noise")
            return np.random.randn(self.chunk_duration * self.sample_rate)
    
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
        
        # Get track metadata (handle both MultiIndex and simple)
        if hasattr(self.tracks, 'iloc'):
            metadata_row = self.tracks.iloc[idx] if idx < len(self.tracks) else None
        else:
            metadata_row = None
        
        # Chunk the audio
        chunks = self._chunk_audio(audio)
        
        # Augment each chunk
        all_chunks = []
        for chunk in chunks:
            all_chunks.extend(self._augment_audio(chunk))
        
        # Extract genre and other metadata
        genre = 'unknown'
        title = f'Track_{track_id}'
        artist = 'Unknown Artist'
        
        if metadata_row is not None:
            if isinstance(self.tracks.columns, pd.MultiIndex):
                # MultiIndex columns
                genre = metadata_row['track', 'genre_top'] if ('track', 'genre_top') in self.tracks.columns else 'unknown'
                title = metadata_row['track', 'title'] if ('track', 'title') in self.tracks.columns else f'Track_{track_id}'
                artist = metadata_row['artist', 'name'] if ('artist', 'name') in self.tracks.columns else 'Unknown Artist'
            else:
                # Simple columns
                for col in ['genre_top', 'genre', 'track_genre_top']:
                    if col in self.tracks.columns:
                        genre = metadata_row[col]
                        break
                for col in ['title', 'track_title']:
                    if col in self.tracks.columns:
                        title = metadata_row[col]
                        break
                for col in ['artist_name', 'artist', 'name']:
                    if col in self.tracks.columns:
                        artist = metadata_row[col]
                        break
        
        return {
            'track_id': track_id,
            'chunks': all_chunks,
            'metadata': {
                'genre_top': genre,
                'duration': self.chunk_duration,
                'title': title,
                'artist': artist
            }
        }


class GTZANDataset(Dataset):
    """GTZAN dataset for validation queries"""
    
    def __init__(self, gtzan_path, sample_rate=16000):
        self.gtzan_path = Path(gtzan_path)
        self.sample_rate = sample_rate
        
        # List all audio files
        self.audio_files = sorted(self.gtzan_path.glob('genres/*/*.wav'))
        
        # If no files found, check alternative paths
        if len(self.audio_files) == 0:
            self.audio_files = sorted(self.gtzan_path.glob('GTZAN-dataset-master/genres/*/*.wav'))
        
        # If still no files, create fake data
        if len(self.audio_files) == 0:
            print("No GTZAN files found. Using fake data for testing.")
            # Create 100 fake entries with random genres
            genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 
                      'jazz', 'metal', 'pop', 'reggae', 'rock']
            self.audio_files = []
            for i, genre in enumerate(genres * 10):
                # Create dummy file paths (won't actually load)
                self.audio_files.append(Path(f'data/gtzan/genres/{genre}/fake_{i:05d}.wav'))
        
        print(f"Found {len(self.audio_files)} GTZAN tracks")
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        
        # Try to load audio
        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        except Exception as e:
            # Return random noise for fake data
            audio = np.random.randn(10 * self.sample_rate)
        
        # Ensure 10 seconds
        target_len = 10 * self.sample_rate
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))
        else:
            audio = audio[:target_len]
        
        # Get genre label from path
        genre = audio_path.parent.name if audio_path.parent.name != 'genres' else 'unknown'
        
        return {
            'audio': audio,
            'genre': genre,
            'path': str(audio_path)
        }