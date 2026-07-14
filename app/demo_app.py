"""
AURA Demo App
"""

import streamlit as st
import random
import pandas as pd
from datetime import datetime

# Page config
st.set_page_config(
    page_title="AURA - Music Discovery",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sample music database (realistic fake data)
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

# Sample queries for quick demo
SAMPLE_QUERIES = [
    "upbeat acoustic folk with melancholic undertones",
    "dark ambient drone music for studying",
    "funky 70s groove with driving bassline",
    "ethereal female vocals with electronic production",
    "energetic electronic dance music",
    "calm relaxing piano for meditation"
]

def get_relevant_tracks(query, k=10):
    """Simulate search based on query keywords."""
    results = []
    query_lower = query.lower()
    
    # Score each track based on keyword matching
    for track in MUSIC_DATABASE:
        score = 0.0
        text = f"{track['title']} {track['artist']} {track['genre']} {track['mood']}".lower()
        
        # Simple keyword matching
        keywords = query_lower.split()
        for word in keywords:
            if word in text:
                score += 0.3
            if word in track['genre'].lower():
                score += 0.2
            if word in track['mood'].lower():
                score += 0.3
        
        # Random slight variation for realism
        score += random.uniform(0, 0.1)
        
        if score > 0:
            results.append({
                **track,
                "similarity": min(1.0, score + 0.2)
            })
    
    # Sort by similarity
    results.sort(key=lambda x: x["similarity"], reverse=True)
    
    # If no results, return random tracks with low similarity
    if not results:
        results = random.sample(MUSIC_DATABASE, min(k, len(MUSIC_DATABASE)))
        for r in results:
            r["similarity"] = random.uniform(0.3, 0.6)
    
    return results[:k]

# HEADER
st.markdown("""
<div style="text-align: center; padding: 1rem 0;">
    <h1 style="font-size: 4rem; margin: 0;">🎵 AURA</h1>
    <p style="font-size: 1.2rem; color: #888;">Discover Music by Vibe</p>
    <p style="font-size: 0.9rem; color: #666; margin-top: -0.5rem;">
        Semantic Music Search Engine
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# SIDEBAR
with st.sidebar:
    st.markdown("### Search Mode")
    search_mode = st.radio(
        "Select query type:",
        ["Text Description", "Audio Upload (Simulated)"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### Sample Queries")
    for sample in SAMPLE_QUERIES:
        if st.button(f"{sample[:35]}...", use_container_width=True, key=sample):
            st.session_state['query_text'] = sample
            st.session_state['run_search'] = True
    
    st.markdown("---")
    st.markdown("### Settings")
    k_results = st.slider("Number of results", 3, 15, 8)
    
    st.markdown("---")
    st.markdown("### Stats")
    st.metric("Total Tracks", f"{len(MUSIC_DATABASE)}")
    st.metric("Query Time", "< 0.1s", delta="fast")
    st.metric("System Status", "🟢 Online")

# MAIN CONTENT
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### Query Input")
    
    query = ""
    
    if search_mode == "Text Description":
        query = st.text_area(
            "Describe the music you're looking for:",
            placeholder="e.g., 'upbeat acoustic folk with a melancholic undertone'",
            height=80,
            value=st.session_state.get('query_text', '')
        )
        
        # Audio upload simulation
        st.info("Tip: Click a sample query in the sidebar or type your own description.")
        
    else:  # Audio Upload
        st.markdown("Upload an audio file (simulated):")
        audio_file = st.file_uploader(
            "Choose an audio file",
            type=["wav", "mp3", "m4a"],
            label_visibility="collapsed"
        )
        
        if audio_file is not None:
            st.success(f"Loaded: {audio_file.name}")
            query = f"Audio: {audio_file.name}"
            st.audio(audio_file, format='audio/wav')
        else:
            st.info("Upload an audio file or use a sample query")
            query = st.session_state.get('query_text', '')

with col2:
    st.markdown("### Search Results")
    
    # Search button
    search_clicked = st.button("Find Music", type="primary", use_container_width=True)
    
    # Check if we should search
    should_search = search_clicked or st.session_state.get('run_search', False)
    
    if should_search and query:
        st.session_state['run_search'] = False
        
        with st.spinner("Searching AURA..."):
            results = get_relevant_tracks(query, k=k_results)
        
        if results:
            st.markdown(f"### Results ({len(results)} tracks)")
            
            for i, track in enumerate(results):
                with st.container():
                    cols = st.columns([1, 4])
                    with cols[0]:
                        # Medal emoji for top 3
                        if i == 0:
                            icon = "🥇"
                        elif i == 1:
                            icon = "🥈"
                        elif i == 2:
                            icon = "🥉"
                        else:
                            icon = f"#{i+1}"
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
                        sim = track['similarity']
                        st.markdown(f"Similarity: {sim:.2f}")
                        st.progress(sim)
                        
                        # Metadata
                        st.caption(f" {track['year']} •  {track['tempo']} BPM")
                    
                    st.divider()
        else:
            st.info("No results found. Try a different query.")
    
    elif should_search and not query:
        st.warning("Please enter a query or click a sample.")

# FOOTER
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>
        <strong>AURA v1.0</strong> • Built with Deep Learning 
        • <a href="https://github.com/sswfml/DLS_project" target="_blank">GitHub</a>
    </p>
    <p style="font-size: 0.8rem;">
        Deep Learning for Search, Summer 2026 • Innopolis University
    </p>
</div>
""", unsafe_allow_html=True)

# RUN SEARCH TRIGGER (for sample queries)
if st.session_state.get('run_search', False) and st.session_state.get('query_text', ''):
    st.rerun()