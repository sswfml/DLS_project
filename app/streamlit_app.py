#!/usr/bin/env python
"""
Streamlit web app for AURA - Music Discovery by Vibe.
"""

import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import tempfile
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.audio_encoder import MERTEncoder
from models.clap_adapter import CLAPAdapter
from search.faiss_index import FAISSSearch
from search.milvus_index import MilvusSearch
from search.hybrid_search import HybridSearch
import yaml


class AuraApp:
    """Main application class for AURA."""

    def __init__(self):
        self.config = self._load_config()
        self.encoder = None
        self.clap = None
        self.index = None
        self.hybrid_search = None
        self._load_models()
        self._load_index()

    def _load_config(self):
        """Load configuration."""
        config_path = Path(__file__).parent.parent / "config" / "config.yaml"
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _load_models(self):
        """Load embedding models."""
        with st.spinner("Loading AURA models..."):
            try:
                self.encoder = MERTEncoder()
            except Exception as e:
                st.warning(f"Could not load MERT: {e}")
            try:
                self.clap = CLAPAdapter()
            except Exception as e:
                st.warning(f"Could not load CLAP: {e}")

    def _load_index(self):
        """Load pre-built search index."""
        with st.spinner("Loading search index..."):
            # Try to load the optimized index
            index_path = Path(__file__).parent.parent / "experiments" / "results" / "iteration2_optimized"
            if index_path.exists():
                try:
                    self.index = MilvusSearch(
                        collection_name="aura_music",
                        dimension=128,
                        host="localhost",
                        port="19530"
                    )
                    # Check if collection exists
                    from pymilvus import utility
                    if utility.has_collection("aura_music"):
                        self.hybrid_search = HybridSearch(
                            self.index,
                            text_corpus=[""] * 1000,  # Placeholder
                            k_rrf=60
                        )
                except Exception as e:
                    st.warning(f"Could not load Milvus index: {e}")
                    self._load_fallback_index()

    def _load_fallback_index(self):
        """Load FAISS index as fallback."""
        index_path = Path(__file__).parent.parent / "experiments" / "results" / "iteration1_baseline"
        if (index_path / "index.faiss").exists():
            try:
                self.index = FAISSSearch(dimension=768)
                self.index.load(str(index_path))
            except Exception as e:
                st.error(f"Could not load any index: {e}")

    def query_audio(self, audio_array, k=10):
        """Query by audio."""
        if self.index is None or self.encoder is None:
            st.error("Models not loaded. Please check setup.")
            return []

        # Generate embedding
        embedding = self.encoder.encode_audio(audio_array)

        # Search
        results, _ = self.index.search(embedding, k=k)
        return results[0] if results else []

    def query_text(self, text, k=10):
        """Query by text description using CLAP."""
        if self.index is None or self.clap is None:
            st.error("CLAP not loaded. Please check setup.")
            return []

        # Generate text embedding
        text_embedding = self.clap.encode_text(text)

        # Search in audio space
        results, _ = self.index.search(text_embedding, k=k)
        return results[0] if results else []

    def query_hybrid(self, audio_array, text, k=10):
        """Hybrid query using both audio and text."""
        if self.hybrid_search is None or self.encoder is None:
            st.warning("Hybrid search not available. Using audio-only.")
            return self.query_audio(audio_array, k)

        # Generate audio embedding
        audio_emb = self.encoder.encode_audio(audio_array)

        # Perform hybrid search
        results = self.hybrid_search.search(audio_emb[0], text, k=k)
        return results


def main():
    """Streamlit app entry point."""
    st.set_page_config(
        page_title="AURA - Music Discovery",
        page_icon="🎵",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize app
    if "aura" not in st.session_state:
        st.session_state.aura = AuraApp()

    aura = st.session_state.aura

    # Header
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h1 style="font-size: 4rem; margin: 0;">🎵 AURA</h1>
        <p style="font-size: 1.2rem; color: #888;">Discover Music by Vibe</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### 🔍 Search Type")
        search_mode = st.radio(
            "Select query mode:",
            ["🎵 Audio / Humming", "📝 Text Description", "🎤 Hybrid (Audio + Text)"]
        )

        st.markdown("---")
        st.markdown("### ⚙️ Settings")
        k_results = st.slider("Number of results", 5, 30, 10)

        st.markdown("---")
        st.markdown("### 💡 Sample Queries")
        sample_texts = aura.config['app']['sample_queries']
        for sample in sample_texts:
            if st.button(f"🔊 {sample[:40]}..."):
                st.session_state.query_text = sample

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 🎤 Query Input")

        if "audio" in search_mode or "Hybrid" in search_mode:
            # Audio upload
            audio_file = st.file_uploader(
                "Upload an audio file (WAV, MP3)",
                type=["wav", "mp3", "m4a"]
            )

            # Or record from microphone
            st.markdown("Or record from microphone:")

            # Streamlit's audio recorder is not built-in; use a workaround
            # For demo, we'll use an upload

            if audio_file is not None:
                # Save to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                    tmp.write(audio_file.read())
                    tmp_path = tmp.name

                # Load audio
                audio, sr = librosa.load(tmp_path, sr=16000, mono=True)
                # Ensure 10 seconds
                target_len = 10 * 16000
                if len(audio) < target_len:
                    audio = np.pad(audio, (0, target_len - len(audio)))
                else:
                    audio = audio[:target_len]

                st.audio(audio_file, format='audio/wav')
                st.success(f"Loaded audio: {len(audio)/16000:.2f} seconds")
                st.session_state.audio_query = audio

        if "Text" in search_mode or "Hybrid" in search_mode:
            text_query = st.text_area(
                "Describe the music you're looking for:",
                placeholder="e.g., 'upbeat acoustic folk with a melancholic undertone'",
                height=80
            )
            if text_query:
                st.session_state.text_query = text_query

    with col2:
        st.markdown("### 🎯 Search Results")

        if st.button("🔮 Find Music", type="primary", use_container_width=True):
            with st.spinner("Searching AURA..."):
                if "audio" in search_mode and "audio_query" in st.session_state:
                    results = aura.query_audio(
                        st.session_state.audio_query,
                        k=k_results
                    )
                elif "Text" in search_mode and "text_query" in st.session_state:
                    results = aura.query_text(
                        st.session_state.text_query,
                        k=k_results
                    )
                elif "Hybrid" in search_mode:
                    if "audio_query" in st.session_state and "text_query" in st.session_state:
                        results = aura.query_hybrid(
                            st.session_state.audio_query,
                            st.session_state.text_query,
                            k=k_results
                        )
                    else:
                        st.warning("Please provide both audio and text for hybrid search.")
                        results = []
                else:
                    st.warning("Please provide a query.")
                    results = []

                # Display results
                if results:
                    for i, result in enumerate(results):
                        with st.container():
                            cols = st.columns([1, 4])
                            with cols[0]:
                                st.markdown(f"### #{i+1}")
                            with cols[1]:
                                if 'metadata' in result and result['metadata']:
                                    meta = result['metadata']
                                    st.markdown(f"**{meta.get('title', 'Unknown Title')}**")
                                    st.markdown(f"*{meta.get('artist', 'Unknown Artist')}*")
                                    st.markdown(f"Genre: {meta.get('genre', 'Unknown')}")
                                else:
                                    st.markdown(f"Track ID: {result.get('id', result.get('track_id', 'Unknown'))}")
                                st.markdown(f"Similarity: {result.get('distance', 1.0):.4f}")
                            st.divider()
                else:
                    st.info("No results found. Try a different query.")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        AURA v1.0 | Built with 🎵 MERT, CLAP, FAISS & Milvus
        <br>
        Deep Learning for Search, Summer 2026 | Innopolis University
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
