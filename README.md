# 🎵 "That's the Vibe" – Mood-Based Music Discovery

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![FAISS](https://img.shields.io/badge/FAISS-1.7.4-green)](https://github.com/facebookresearch/faiss)
[![Milvus](https://img.shields.io/badge/Milvus-2.3.0-blue)](https://milvus.io/)

## 📖 Project Overview

This project implements a **semantic music search engine** that retrieves tracks based on their acoustic "vibe" rather than metadata. Users can query using:
- 🎵 **Audio snippets** (10 seconds of humming or a song)
- 📝 **Text descriptions** (e.g., "upbeat acoustic folk with melancholic undertones")

The system uses **deep learning embeddings** from state-of-the-art music understanding models (MERT, CLAP) and evaluates multiple vector search strategies (FAISS-HNSW, Milvus-IVF-PQ) to find the optimal trade-off between speed, memory, and accuracy.

## 🎯 Value Proposition

- **For DJs & Producers**: Discover sample tracks and remix stems based on sonic similarity
- **For Music Therapists**: Curate emotionally resonant playlists
- **For Casual Listeners**: Break recommendation bubbles and explore music by feeling
- **For Game/Film Designers**: Source background music matching scene emotions

## 🏗️ System Architecture

![System Architecture](presentation/assets/architecture.png)

## 📊 Dataset

- **FMA (Free Music Archive) Large**: 106,000 tracks, sliced into 10-second chunks
- **GTZAN**: 1,000 labeled tracks for validation
- **Total**: ~150,000 chunks (augmented to 500,000+)
- **Embeddings**: 768-dim from MERT model

## 🔬 Experimental Iterations

| Iteration | Model | Index | Dim | Recall@10 | QPS | Memory |
|-----------|-------|-------|-----|-----------|-----|--------|
| 1 (Baseline) | MERT | FAISS-HNSW | 768 | 0.92 | 35 | 450MB |
| 2 (Optimized) | MERT + PCA + Quantization | Milvus-IVF-PQ | 128 (int8) | 0.87 | 210 | 85MB |
| 3 (Multimodal) | CLAP (text-to-audio) | Milvus-IVF-PQ | 512 | 0.68 | 190 | 110MB |

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt