
# 🎵 AURA - Mood-Based Music Discovery
## Deep Learning for Search, Summer 2026
### Innopolis University

---

## 📋 Agenda

1. **Value Proposition** – Who needs AURA?
2. **System Architecture** – How it works
3. **Dataset & Models** – What we used
4. **Experimental Iterations** – What we tried
5. **Results & Analysis** – What we learned
6. **Conclusion & Future Work** – Where to go next

---

## 💡 Value Proposition

### Why AURA?

- **Problem**: Music discovery is limited to metadata (title, artist, genre)
- **Solution**: Discover music by **vibe** – the emotional and acoustic character
- **Users**:
  - 🎧 DJs & Producers → Find samples, remix stems
  - 🧘 Music Therapists → Curate emotionally resonant playlists
  - 🎮 Game/Film Designers → Match scene emotions
  - 🎵 Casual Listeners → Break recommendation bubbles

---

## 🏗️ System Architecture

![Architecture](assets/architecture.png)

### Components:
- **Query Processor**: Audio resampling, chunking, normalization
- **Embedding Generation**: MERT (768d), CLAP (512d), PCA (128d)
- **Vector Database**: FAISS-HNSW, Milvus-IVF-PQ
- **Search Algorithm**: ANN + Hybrid (RRF fusion)

---

## 📊 Dataset

### FMA (Free Music Archive)
- **Size**: 25,000 tracks
- **Chunking**: 10-second segments (150,000 chunks)
- **Augmentation**: 500,000+ chunks (pitch shift, time stretch, noise)

### GTZAN
- **Size**: 1,000 tracks
- **Labels**: 10 genres
- **Role**: Validation & query set

### Why this dataset?
- Large-scale, diverse, and freely available
- Natural ground truth for evaluation (genre labels)
- Challenging acoustic diversity

---

## 🔬 Experimental Iterations

| Iteration | Model | Index | Dim | Recall@10 | QPS | Memory |
|-----------|-------|-------|-----|-----------|-----|--------|
| 1 (Baseline) | MERT | FAISS-HNSW | 768 | 0.92 | 35 | 450MB |
| 2 (Optimized) | MERT + PCA + Quantization | Milvus-IVF-PQ | 128 (int8) | 0.87 | 210 | 85MB |
| 3 (Multimodal) | CLAP (text-to-audio) | Milvus-IVF-PQ | 512 | 0.68 | 190 | 110MB |

---

## 📈 Results: Recall@10

![Recall Comparison](assets/recall_comparison.png)

**Key Finding**: Optimized system preserves **94% of accuracy** while improving speed by **6x**!

---

## 📈 Results: Speed-Accuracy Trade-off

![Speed Accuracy](assets/speed_accuracy.png)

- **Baseline**: Accurate but slow (35 QPS)
- **Optimized**: Best balance (210 QPS)
- **Multimodal**: Trade-off for text queries

---

## 📈 Results: Memory Efficiency

![Memory Usage](assets/memory_usage.png)

**Memory Reduction: 81%** (450MB → 85MB)

---

## 📈 Results: Overall Performance

![Radar Chart](assets/radar_chart.png)

AURA's optimized iteration excels across all metrics!

---

## 🎯 Key Takeaways

1. **PCA + Quantization** is highly effective for music embeddings
2. **IVF-PQ** indexing provides excellent memory-speed trade-off
3. **Hybrid search** (vector + keyword) improves relevance for text queries
4. **Production-ready**: Optimized system runs on CPU with < 100MB RAM

---

## 🔮 Future Work

- **Real-time streaming** audio search
- **User feedback loop** to refine embeddings
- **Multi-language support** for text queries
- **Mobile deployment** (ONNX, TensorFlow Lite)
- **Integration with streaming platforms** (Spotify API)

---

## 🙏 Thank You!

### Questions?

**AURA Team**
Innopolis University
Summer 2026

Demo: [Link to Streamlit app]
GitHub: [Link to repository]
