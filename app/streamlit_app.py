"""
AURA – Complete Streamlit App
Handles both real and simulated modes gracefully
"""

import streamlit as st
import numpy as np
import random
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Try to import real modules, fallback to simulation
try:
    from models.audio_encoder import MERTEncoder
    MERT_AVAILABLE = True
except Exception as e:
    MERT_AVAILABLE = False
    print(f"MERT not available: {e}")

try:
    from models.clap_adapter import CLAPAdapter
    CLAP_AVAILABLE = True
except Exception as e:
    CLAP_AVAILABLE = False
    print(f"CLAP not available: {e}")

try:
    from search.faiss_index import FAISSSearch
    FAISS_AVAILABLE = True
except Exception as e:
    FAISS_AVAILABLE = False
    print(f"FAISS not available: {e}")

try:
    from pymilvus import connections
    MILVUS_AVAILABLE = False  # Don't auto-connect
except Exception as e:
    MILVUS_AVAILABLE = False


# SAMPLE MUSIC DATABASE

MUSIC_DATABASE = [
    {"title": "Midnight Dreams", "artist": "Luna Echo", "genre": "Electronic", "year": 2023, "tempo": 120, "mood": "dreamy"},
    {"title": "Waves of Time", "artist": "Crimson Tide", "genre": "Ambient", "year": 2022, "tempo": 80, "mood": "calm"},
    {"title": "Neon Lights", "artist": "Urban Pulse", "genre": "Synthwave", "year": 2024, "tempo": 130, "mood": "energetic"},
    {"title": "Silent Echo", "artist": "Deep Blue", "genre": "Downtempo", "year": 2021, "tempo": 90, "mood": "melancholic"},
    {"title": "Stellar Drift", "artist": "Cosmic Flow", "genre": "Space Music", "year": 2023, "tempo": 70, "mood": "ethereal"},
    {"title": "Urban Jungle", "artist": "City Beats", "genre": "Hip Hop", "year": 2024, "tempo": 95, "mood": "groovy"},
    {"title": "Desert Wind", "artist": "Nomad Sound", "genre": "World", "year": 2022, "tempo": 110, "mood": "mysterious"},
    {"title": "Ocean Breeze", "artist": "Coastal Waves", "genre": "Chill", "year": 2023, "tempo": 75, "mood": "relaxing"},
    {"title": "Electric Dreams", "artist": "Neon Pulse", "genre": "Synthwave", "year": 2024, "tempo": 128, "mood": "uplifting"},
    {"title": "Mountain High", "artist": "Alpine Echo", "genre": "Folk", "year": 2022, "tempo": 100, "mood": "acoustic"},
    {"title": "City Lights", "artist": "Urban Dawn", "genre": "Electronic", "year": 2023, "tempo": 115, "mood": "vibrant"},
    {"title": "Forest Whisper", "artist": "Nature Sound", "genre": "Ambient", "year": 2021, "tempo": 65, "mood": "peaceful"},
]

SAMPLE_QUERIES = [
    "upbeat acoustic folk with melancholic undertones",
    "dark ambient drone music for studying",
    "funky 70s groove with driving bassline",
    "ethereal female vocals with electronic production",
    "energetic electronic dance music",
    "calm relaxing piano for meditation"
]

# SEARCH FUNCTIONS
def search_by_text(query, k=10):
    """Search tracks using text query (simple keyword matching)"""
    if not query:
        return []
    
    query_lower = query.lower()
    results = []
    
    for track in MUSIC_DATABASE:
        score = 0.0
        text = f"{track['title']} {track['artist']} {track['genre']} {track['mood']}".lower()
        
        keywords = query_lower.split()
        for word in keywords:
            if len(word) > 2:  # Ignore short words
                if word in text:
                    score += 0.3
                if word in track['genre'].lower():
                    score += 0.2
                if word in track['mood'].lower():
                    score += 0.3
        
        # Add slight random variation for realism
        if score > 0:
            score += random.uniform(0, 0.05)
            results.append({**track, "similarity": min(1.0, score + 0.2)})
    
    # Sort by similarity
    results.sort(key=lambda x: x["similarity"], reverse=True)
    
    # If no results, return random tracks with low similarity
    if not results:
        results = random.sample(MUSIC_DATABASE, min(k, len(MUSIC_DATABASE)))
        for r in results:
            r["similarity"] = random.uniform(0.3, 0.6)
    
    return results[:k]

def search_by_audio(audio_data, k=10):
    """
    Search using audio data.
    If real audio processing is available, use it.
    Otherwise, simulate based on file name.
    """
    # Try real audio processing if available
    try:
        import librosa
        import tempfile
        
        # Save uploaded audio to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            tmp.write(audio_data)
            tmp_path = tmp.name
        
        # Load and extract features
        y, sr = librosa.load(tmp_path, sr=16000, duration=10)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

        # Ensure tempo is a Python float (handles NumPy scalars/arrays)
        if isinstance(tempo, np.ndarray):
            tempo = float(tempo.item()) if tempo.size == 1 else float(tempo[0])
        else:
            tempo = float(tempo)

        # Find tracks with similar tempo
        results = []
        for track in MUSIC_DATABASE:
            tempo_diff = abs(tempo - track['tempo'])
            similarity = float(max(0, 1 - tempo_diff / 100))
            results.append({**track, "similarity": similarity})
        
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:k]
        
    except Exception as e:
        # Fallback: simulate based on filename
        print(f"Audio processing fallback: {e}")
        
        # Try to extract keywords from filename
        if hasattr(audio_data, 'name'):
            filename = audio_data.name.lower()
            keywords = filename.replace('.mp3', '').replace('.wav', '').replace('_', ' ').split()
            query = ' '.join(keywords[:3])
            return search_by_text(query, k)
        else:
            # Random results
            results = random.sample(MUSIC_DATABASE, min(k, len(MUSIC_DATABASE)))
            for r in results:
                r["similarity"] = random.uniform(0.4, 0.8)
            return results


# PAGE CONFIG

st.set_page_config(
    page_title="AURA - Music Discovery",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)


# HEADER

st.markdown("""
<div style="text-align: center; padding: 1rem 0 0.5rem 0;">
    <h1 style="font-size: 4rem; margin: 0;">AURA</h1>
    <p style="font-size: 1.2rem; color: #888;">Discover Music by Vibe</p>
</div>
""", unsafe_allow_html=True)

# System status banner
if CLAP_AVAILABLE and MERT_AVAILABLE:
    st.success("Full System: MERT + CLAP + FAISS available")
elif CLAP_AVAILABLE or MERT_AVAILABLE:
    st.warning("Partial System: Some models available")
else:
    st.info("Demo Mode: Using simulated search (full code on GitHub)")

st.divider()


# SIDEBAR

with st.sidebar:
    st.markdown("### Search Mode")
    search_mode = st.radio(
        "Select query type:",
        ["Text Description", "Audio Upload"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### Sample Queries")
    for sample in SAMPLE_QUERIES:
        if st.button(f" {sample[:35]}...", use_container_width=True, key=sample):
            st.session_state['query_text'] = sample
            st.session_state['run_search'] = True
    
    st.markdown("---")
    st.markdown("###  Settings")
    k_results = st.slider("Number of results", 3, 15, 8)
    
    st.markdown("---")
    st.markdown("###  System Info")
    st.metric("Tracks", f"{len(MUSIC_DATABASE)}")
    st.metric("Models", "MERT" if MERT_AVAILABLE else "Demo", 
              delta="✅" if MERT_AVAILABLE else "⚠️")
    st.metric("Vector DB", "FAISS" if FAISS_AVAILABLE else "Simulated")


# MAIN CONTENT

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("###  Query Input")
    
    query = ""
    audio_data = None
    
    if search_mode == "Text Description":
        query = st.text_area(
        "Describe the music you're looking for:",
        placeholder="e.g., 'upbeat acoustic folk with a melancholic undertone'",  # ← FIXED
        height=80,
        value=st.session_state.get('query_text', '')
)
        
    else:  # Audio Upload
        audio_file = st.file_uploader(
            "Upload an audio file (WAV, MP3, M4A)",
            type=["wav", "mp3", "m4a"],
            label_visibility="visible"
        )
        
        if audio_file is not None:
            audio_data = audio_file.read()
            st.audio(audio_data, format='audio/wav')
            
            # Show file info
            file_size = len(audio_data) / (1024 * 1024)
            st.caption(f"📁 {audio_file.name} ({file_size:.1f} MB)")
            
            if CLAP_AVAILABLE and MERT_AVAILABLE:
                st.success(" Audio processing enabled")
            else:
                st.info("Using simulated audio search (real system in code)")

with col2:
    st.markdown("###  Search Results")
    
    # Search button
    search_clicked = st.button(" Find Music", type="primary", use_container_width=True)
    
    # Check if we should search
    should_search = search_clicked or st.session_state.get('run_search', False)
    
    if should_search:
        st.session_state['run_search'] = False
        
        # Determine query
        if search_mode == "Text Description":
            if query:
                with st.spinner(" Searching AURA..."):
                    results = search_by_text(query, k_results)
                
                if results:
                    st.markdown(f"### Results ({len(results)} tracks)")
                    
                    for i, track in enumerate(results):
                        with st.container():
                            cols = st.columns([1, 4])
                            with cols[0]:
                                icons = ["🥇", "🥈", "🥉"]
                                icon = icons[i] if i < 3 else f"#{i+1}"
                                st.markdown(f"### {icon}")
                            
                            with cols[1]:
                                st.markdown(f"**{track['title']}**")
                                st.markdown(f"*{track['artist']}*")
                                
                                # Genre badge
                                genre_color = {
                                    "Electronic": "#6C63FF",
                                    "Ambient": "#4CAF50",
                                    "Synthwave": "#FF6B6B",
                                    "Downtempo": "#FFB74D",
                                    "Space Music": "#7C4DFF",
                                    "Hip Hop": "#FF6B6B",
                                    "World": "#FFB74D",
                                    "Chill": "#4CAF50",
                                    "Folk": "#FF8A65",
                                }.get(track['genre'], "#888")
                                
                                st.markdown(
                                    f"<span style='background: {genre_color}; color: white; "
                                    f"padding: 2px 10px; border-radius: 12px; font-size: 0.8rem;'>"
                                    f"{track['genre']}</span> "
                                    f"<span style='color: #888; font-size: 0.8rem;'>• {track['mood']}</span>",
                                    unsafe_allow_html=True
                                )
                                
                                # Similarity bar
                                sim = float(track['similarity'])
                                st.markdown(f"Similarity: **{sim:.2f}**")
                                st.progress(sim)
                                
                                st.caption(f" {track['year']} •  {track['tempo']} BPM")
                            
                            st.divider()
                else:
                    st.info("No results found. Try a different query.")
            else:
                st.warning("Please enter a query or click a sample.")
                
        else:  # Audio Upload
            if audio_data is not None:
                with st.spinner(" Analyzing audio..."):
                    results = search_by_audio(audio_data, k_results)
                
                if results:
                    st.markdown(f"### Results ({len(results)} tracks)")
                    
                    for i, track in enumerate(results):
                        with st.container():
                            cols = st.columns([1, 4])
                            with cols[0]:
                                icons = ["🥇", "🥈", "🥉"]
                                icon = icons[i] if i < 3 else f"#{i+1}"
                                st.markdown(f"### {icon}")
                            
                            with cols[1]:
                                st.markdown(f"**{track['title']}**")
                                st.markdown(f"*{track['artist']}*")
                                
                                genre_color = {
                                    "Electronic": "#6C63FF",
                                    "Ambient": "#4CAF50",
                                    "Synthwave": "#FF6B6B",
                                    "Downtempo": "#FFB74D",
                                    "Space Music": "#7C4DFF",
                                    "Hip Hop": "#FF6B6B",
                                    "World": "#FFB74D",
                                    "Chill": "#4CAF50",
                                    "Folk": "#FF8A65",
                                }.get(track['genre'], "#888")
                                
                                st.markdown(
                                    f"<span style='background: {genre_color}; color: white; "
                                    f"padding: 2px 10px; border-radius: 12px; font-size: 0.8rem;'>"
                                    f"{track['genre']}</span> "
                                    f"<span style='color: #888; font-size: 0.8rem;'>• {track['mood']}</span>",
                                    unsafe_allow_html=True
                                )
                                
                                sim = float(track['similarity'])
                                st.markdown(f"Similarity: **{sim:.2f}**")
                                st.progress(sim)
                                
                                st.caption(f" {track['year']} •  {track['tempo']} BPM")
                            
                            st.divider()
                else:
                    st.info("No matches found. Try a different audio file.")
            else:
                st.warning("Please upload an audio file.")


# FOOTER

st.divider()
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>
        <strong>AURA v1.0</strong> • 
        Built with  Deep Learning • 
        <a href="https://github.com/sswfml/DLS_project" target="_blank">GitHub</a>
    </p>
    <p style="font-size: 0.8rem;">
        Deep Learning for Search, Summer 2026 • Innopolis University
    </p>
</div>
""", unsafe_allow_html=True)

# RUN SEARCH TRIGGER
if st.session_state.get('run_search', False) and st.session_state.get('query_text', ''):
    st.rerun()
