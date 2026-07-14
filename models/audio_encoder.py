import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModel, AutoProcessor
import librosa
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class MERTEncoder:
    """MERT model for music embedding generation"""
    
    def __init__(self, model_name="m-a-p/MERT-v1-95M", device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        
        print(f"Loading MERT model on {self.device}...")
        self.model = AutoModel.from_pretrained(
            model_name, 
            trust_remote_code=True
        ).to(self.device)
        self.model.eval()
        
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.sample_rate = 24000
        self.embedding_dim = 768
    
    @torch.no_grad()
    def encode_audio(self, audio_array):
        """Generate embedding for a single audio chunk"""
        if len(audio_array.shape) == 1:
            audio_array = audio_array.reshape(1, -1)
        
        # Process audio
        inputs = self.processor(
            audio_array,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        # Generate embeddings
        outputs = self.model(**inputs)
        
        # Mean pooling over time dimension
        embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return embeddings.cpu().numpy()
    
    @torch.no_grad()
    def encode_batch(self, audio_batch, batch_size=32):
        """Generate embeddings for a batch of audio chunks"""
        all_embeddings = []
        
        for i in tqdm(range(0, len(audio_batch), batch_size), desc="Encoding audio"):
            batch = audio_batch[i:i+batch_size]
            
            # Pad batch to same length
            max_len = max(len(a) for a in batch)
            padded = np.array([
                np.pad(a, (0, max_len - len(a))) for a in batch
            ])
            
            embeddings = self.encode_audio(padded)
            all_embeddings.append(embeddings)
        
        return np.vstack(all_embeddings) if all_embeddings else np.array([])

class LightweightAudioEncoder(nn.Module):
    """Lightweight CNN encoder for optimized inference"""
    
    def __init__(self, input_dim=128, embedding_dim=128):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            
            nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, embedding_dim)
        )
        
        self.embedding_dim = embedding_dim
    
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        return self.encoder(x)

class EmbeddingOptimizer:
    """Optimize embeddings via PCA and quantization"""
    
    def __init__(self, original_dim, target_dim):
        self.original_dim = original_dim
        self.target_dim = target_dim
        self.pca = None
        self.scaler = None
    
    def fit_pca(self, embeddings, n_components=None):
        """Fit PCA on a subset of embeddings"""
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        
        self.scaler = StandardScaler()
        scaled = self.scaler.fit_transform(embeddings)
        
        n_components = n_components or self.target_dim
        self.pca = PCA(n_components=n_components)
        self.pca.fit(scaled)
        
        print(f"PCA explained variance: {self.pca.explained_variance_ratio_.sum():.4f}")
        return self
    
    def transform(self, embeddings):
        """Apply PCA transformation"""
        if self.pca is None:
            raise ValueError("PCA not fitted. Call fit_pca first.")
        
        scaled = self.scaler.transform(embeddings)
        reduced = self.pca.transform(scaled)
        return reduced
    
    def quantize(self, embeddings, dtype=np.int8):
        """Quantize embeddings to reduce memory footprint"""
        if dtype == np.int8:
            # Scale to [-128, 127] range
            min_val = embeddings.min(axis=0)
            max_val = embeddings.max(axis=0)
            
            # Avoid division by zero
            range_val = max_val - min_val
            range_val[range_val == 0] = 1
            
            quantized = ((embeddings - min_val) / range_val) * 255 - 128
            quantized = np.round(quantized).astype(np.int8)
            
            # Store scaling parameters for dequantization
            self.quant_params = {'min': min_val, 'range': range_val}
            
            return quantized
        else:
            return embeddings.astype(dtype)
    
    def dequantize(self, quantized):
        """Dequantize embeddings back to float"""
        if not hasattr(self, 'quant_params'):
            raise ValueError("Quantization parameters not stored.")
        
        return ((quantized.astype(np.float32) + 128) / 255) * self.quant_params['range'] + self.quant_params['min']
