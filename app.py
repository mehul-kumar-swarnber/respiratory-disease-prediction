import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os
import tempfile
from utils import predict_single_audio, validate_audio_file, load_model_safely

st.set_page_config(
    page_title="RespiAI — Respiratory Analysis",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@300;400;500;600&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
/* ── RESET & BASE ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, .stApp {
    font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    background: transparent !important;
}

/* ── THEME VARIABLES ── */
:root {
    --glass-bg-light: rgba(255,255,255,0.55);
    --glass-bg-dark: rgba(15,15,25,0.55);
    --glass-border-light: rgba(255,255,255,0.75);
    --glass-border-dark: rgba(255,255,255,0.10);
    --blur: blur(24px) saturate(180%);
    --accent: #3b82f6;
    --accent-glow: rgba(59,130,246,0.35);
    --accent2: #06b6d4;
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
    --radius-xl: 20px;
    --radius-lg: 14px;
    --radius-md: 10px;
    --shadow-glass: 0 8px 40px rgba(0,0,0,0.12), 0 1.5px 0 rgba(255,255,255,0.6) inset;
    --shadow-glass-dark: 0 8px 40px rgba(0,0,0,0.45), 0 1.5px 0 rgba(255,255,255,0.07) inset;
}

/* ── ANIMATED GRADIENT BACKGROUND ── */
.stApp {
    background: linear-gradient(135deg,
        #e0f2fe 0%,
        #f0fdf4 25%,
        #fdf4ff 50%,
        #fff7ed 75%,
        #e0f2fe 100%
    ) !important;
    background-size: 400% 400% !important;
    animation: gradientShift 18s ease infinite !important;
    min-height: 100vh;
}

@media (prefers-color-scheme: dark) {
    .stApp {
        background: linear-gradient(135deg,
            #0a0f1e 0%,
            #0d1a2e 25%,
            #1a0d2e 50%,
            #0d1a1a 75%,
            #0a0f1e 100%
        ) !important;
    }
}

@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* ── FLOATING ORBS ── */
.stApp::before {
    content: '';
    position: fixed;
    top: -20%;
    left: -10%;
    width: 600px;
    height: 600px;
    background: radial-gradient(circle, rgba(59,130,246,0.15) 0%, transparent 70%);
    border-radius: 50%;
    animation: floatOrb1 20s ease-in-out infinite;
    pointer-events: none;
    z-index: 0;
}
.stApp::after {
    content: '';
    position: fixed;
    bottom: -10%;
    right: -5%;
    width: 500px;
    height: 500px;
    background: radial-gradient(circle, rgba(16,185,129,0.12) 0%, transparent 70%);
    border-radius: 50%;
    animation: floatOrb2 25s ease-in-out infinite;
    pointer-events: none;
    z-index: 0;
}
@keyframes floatOrb1 {
    0%,100% { transform: translate(0,0) scale(1); }
    33%      { transform: translate(5%,3%) scale(1.05); }
    66%      { transform: translate(-3%,5%) scale(0.97); }
}
@keyframes floatOrb2 {
    0%,100% { transform: translate(0,0) scale(1); }
    50%      { transform: translate(-4%,-3%) scale(1.08); }
}

/* ── GLASS CARD MIXIN ── */
.glass-card {
    background: var(--glass-bg-light);
    backdrop-filter: var(--blur);
    -webkit-backdrop-filter: var(--blur);
    border: 1px solid var(--glass-border-light);
    border-radius: var(--radius-xl);
    box-shadow: var(--shadow-glass);
    padding: 1.75rem 2rem;
    margin-bottom: 1.25rem;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
    position: relative;
    overflow: hidden;
}
.glass-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.8), transparent);
}
@media (prefers-color-scheme: dark) {
    .glass-card {
        background: var(--glass-bg-dark);
        border-color: var(--glass-border-dark);
        box-shadow: var(--shadow-glass-dark);
    }
    .glass-card::before {
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.12), transparent);
    }
}

/* ── HEADER ── */
.respi-header {
    background: rgba(255,255,255,0.4);
    backdrop-filter: var(--blur);
    -webkit-backdrop-filter: var(--blur);
    border: 1px solid rgba(255,255,255,0.7);
    border-radius: var(--radius-xl);
    padding: 2.5rem 2.5rem 2rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 40px rgba(59,130,246,0.08), var(--shadow-glass);
}
.respi-header::after {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 350px;
    height: 350px;
    background: radial-gradient(circle, rgba(59,130,246,0.08) 0%, transparent 65%);
    border-radius: 50%;
}
.respi-header h1 {
    font-family: 'DM Sans', sans-serif;
    font-size: 2.2rem;
    font-weight: 600;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #1e40af, #0891b2, #059669);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem;
    line-height: 1.2;
}
.respi-header p {
    font-size: 1rem;
    color: rgba(30,58,138,0.65);
    font-weight: 400;
    max-width: 520px;
    line-height: 1.6;
}
@media (prefers-color-scheme: dark) {
    .respi-header {
        background: rgba(15,15,30,0.5);
        border-color: rgba(255,255,255,0.08);
    }
    .respi-header h1 {
        background: linear-gradient(135deg, #93c5fd, #67e8f9, #6ee7b7);
        -webkit-background-clip: text;
        background-clip: text;
    }
    .respi-header p { color: rgba(186,230,253,0.6); }
}

/* ── BADGE PILL ── */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 14px;
    border-radius: 100px;
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.02em;
    border: 1px solid;
}
.badge-blue {
    background: rgba(59,130,246,0.12);
    color: #1d4ed8;
    border-color: rgba(59,130,246,0.25);
}
.badge-green {
    background: rgba(16,185,129,0.12);
    color: #065f46;
    border-color: rgba(16,185,129,0.25);
}
.badge-amber {
    background: rgba(245,158,11,0.12);
    color: #92400e;
    border-color: rgba(245,158,11,0.25);
}
@media (prefers-color-scheme: dark) {
    .badge-blue  { color: #93c5fd; border-color: rgba(59,130,246,0.35); }
    .badge-green { color: #6ee7b7; border-color: rgba(16,185,129,0.35); }
    .badge-amber { color: #fde68a; border-color: rgba(245,158,11,0.35); }
}

/* ── METRIC CARD ── */
.metric-glass {
    background: rgba(255,255,255,0.6);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.8);
    border-radius: var(--radius-lg);
    padding: 1.25rem 1.5rem;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.06);
    transition: transform 0.2s;
}
.metric-glass:hover { transform: translateY(-2px); }
.metric-glass .metric-label {
    font-size: 0.75rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: rgba(100,116,139,0.9);
    margin-bottom: 0.4rem;
}
.metric-glass .metric-value {
    font-size: 2rem;
    font-weight: 600;
    letter-spacing: -0.04em;
    color: #1e293b;
}
.metric-glass .metric-sub {
    font-size: 0.8rem;
    color: rgba(100,116,139,0.75);
    margin-top: 0.2rem;
}
@media (prefers-color-scheme: dark) {
    .metric-glass {
        background: rgba(15,15,30,0.55);
        border-color: rgba(255,255,255,0.08);
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .metric-glass .metric-value { color: #e2e8f0; }
    .metric-glass .metric-label { color: rgba(148,163,184,0.8); }
}

/* ── PREDICTION RESULT ── */
.prediction-result {
    background: linear-gradient(135deg, rgba(255,255,255,0.7), rgba(255,255,255,0.4));
    backdrop-filter: var(--blur);
    -webkit-backdrop-filter: var(--blur);
    border-radius: var(--radius-xl);
    padding: 2rem;
    border: 1px solid rgba(255,255,255,0.8);
    box-shadow: 0 12px 48px rgba(59,130,246,0.10), var(--shadow-glass);
    position: relative;
    overflow: hidden;
}
.prediction-result::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #3b82f6, #06b6d4, #10b981);
    border-radius: 3px 3px 0 0;
}
.prediction-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: rgba(100,116,139,0.8);
    margin-bottom: 0.5rem;
}
.prediction-disease {
    font-size: 2.4rem;
    font-weight: 700;
    letter-spacing: -0.04em;
    line-height: 1;
    background: linear-gradient(135deg, #1e40af, #0891b2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.6rem;
}
.prediction-disease.healthy {
    background: linear-gradient(135deg, #059669, #0d9488);
    -webkit-background-clip: text;
    background-clip: text;
}
.confidence-bar-track {
    height: 8px;
    background: rgba(203,213,225,0.4);
    border-radius: 100px;
    overflow: hidden;
    margin: 0.75rem 0;
}
.confidence-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #3b82f6, #06b6d4);
    border-radius: 100px;
    transition: width 1s cubic-bezier(0.4,0,0.2,1);
}
.confidence-bar-fill.healthy {
    background: linear-gradient(90deg, #10b981, #0d9488);
}
@media (prefers-color-scheme: dark) {
    .prediction-result {
        background: linear-gradient(135deg, rgba(15,20,40,0.7), rgba(15,20,40,0.4));
        border-color: rgba(255,255,255,0.08);
    }
    .prediction-disease {
        background: linear-gradient(135deg, #93c5fd, #67e8f9);
        -webkit-background-clip: text;
        background-clip: text;
    }
    .prediction-disease.healthy {
        background: linear-gradient(135deg, #6ee7b7, #5eead4);
        -webkit-background-clip: text;
        background-clip: text;
    }
}

/* ── DISCLAIMER CARD ── */
.disclaimer-glass {
    background: rgba(254,252,232,0.6);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(251,191,36,0.3);
    border-radius: var(--radius-lg);
    padding: 1rem 1.25rem;
    margin: 1rem 0;
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
    font-size: 0.85rem;
    color: #78350f;
    line-height: 1.55;
}
.disclaimer-glass .disc-icon {
    font-size: 1.1rem;
    flex-shrink: 0;
    margin-top: 1px;
}
@media (prefers-color-scheme: dark) {
    .disclaimer-glass {
        background: rgba(30,20,5,0.5);
        border-color: rgba(251,191,36,0.2);
        color: #fde68a;
    }
}

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.45) !important;
    backdrop-filter: var(--blur) !important;
    -webkit-backdrop-filter: var(--blur) !important;
    border-right: 1px solid rgba(255,255,255,0.6) !important;
}
section[data-testid="stSidebar"] > div {
    background: transparent !important;
}
@media (prefers-color-scheme: dark) {
    section[data-testid="stSidebar"] {
        background: rgba(5,8,20,0.6) !important;
        border-right-color: rgba(255,255,255,0.06) !important;
    }
}

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.5) !important;
    backdrop-filter: blur(16px) !important;
    border: 1.5px dashed rgba(59,130,246,0.4) !important;
    border-radius: var(--radius-xl) !important;
    padding: 1.5rem !important;
    transition: border-color 0.2s, background 0.2s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(59,130,246,0.7) !important;
    background: rgba(239,246,255,0.6) !important;
}
@media (prefers-color-scheme: dark) {
    [data-testid="stFileUploader"] {
        background: rgba(15,20,40,0.5) !important;
        border-color: rgba(59,130,246,0.3) !important;
    }
}

/* ── BUTTONS ── */
.stButton > button {
    background: linear-gradient(135deg, rgba(59,130,246,0.9), rgba(6,182,212,0.9)) !important;
    color: white !important;
    border: none !important;
    border-radius: 100px !important;
    padding: 0.6rem 1.6rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.01em !important;
    transition: transform 0.15s, box-shadow 0.15s !important;
    box-shadow: 0 4px 15px rgba(59,130,246,0.35) !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 25px rgba(59,130,246,0.5) !important;
}
.stButton > button:active {
    transform: translateY(0) scale(0.98) !important;
}

/* ── EXPANDER ── */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.4) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255,255,255,0.6) !important;
    border-radius: var(--radius-lg) !important;
    overflow: hidden !important;
}
@media (prefers-color-scheme: dark) {
    [data-testid="stExpander"] {
        background: rgba(15,20,40,0.4) !important;
        border-color: rgba(255,255,255,0.08) !important;
    }
}

/* ── AUDIO PLAYER ── */
audio {
    width: 100%;
    border-radius: var(--radius-lg) !important;
    background: rgba(255,255,255,0.5) !important;
}

/* ── PLOTLY CHARTS ── */
.js-plotly-plot {
    border-radius: var(--radius-lg) !important;
    overflow: hidden !important;
}

/* ── DATAFRAME ── */
[data-testid="stDataFrame"] {
    border-radius: var(--radius-lg) !important;
    overflow: hidden !important;
    border: 1px solid rgba(255,255,255,0.5) !important;
    backdrop-filter: blur(12px) !important;
}

/* ── SPINNERS / ALERTS ── */
[data-testid="stAlert"] {
    background: rgba(255,255,255,0.5) !important;
    backdrop-filter: blur(12px) !important;
    border-radius: var(--radius-lg) !important;
    border: 1px solid rgba(255,255,255,0.6) !important;
}

/* ── TYPOGRAPHY ── */
h1,h2,h3,h4,h5,h6 {
    font-family: 'DM Sans', sans-serif !important;
    letter-spacing: -0.02em !important;
}
.section-title {
    font-size: 1.1rem;
    font-weight: 600;
    letter-spacing: -0.02em;
    color: #1e293b;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
@media (prefers-color-scheme: dark) {
    .section-title { color: #e2e8f0; }
}

/* ── SIDEBAR STAT PILLS ── */
.sidebar-stat {
    background: rgba(255,255,255,0.5);
    border: 1px solid rgba(255,255,255,0.7);
    border-radius: var(--radius-md);
    padding: 0.6rem 0.9rem;
    font-size: 0.82rem;
    margin: 0.3rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    color: #334155;
    backdrop-filter: blur(8px);
}
@media (prefers-color-scheme: dark) {
    .sidebar-stat {
        background: rgba(15,20,40,0.5);
        border-color: rgba(255,255,255,0.08);
        color: #94a3b8;
    }
}

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: rgba(148,163,184,0.4);
    border-radius: 100px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(148,163,184,0.7);
}

/* ── HIDE STREAMLIT CHROME ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
    max-width: 1100px !important;
}
</style>
""", unsafe_allow_html=True)


# ── DISEASE INFO ──
disease_info = {
    'COPD': ('Chronic Obstructive Pulmonary Disease', 'Blocks normal airflow and makes breathing difficult.', '🫁'),
    'Healthy': ('Normal Respiratory Sounds', 'No abnormalities detected in the audio sample.', '✅'),
    'URTI': ('Upper Respiratory Tract Infection', 'Affects nose, throat, and upper airways.', '🤧'),
    'Bronchiectasis': ('Bronchiectasis', 'Abnormal widening of the bronchial tubes.', '🔬'),
    'Pneumonia': ('Pneumonia', 'Infection inflaming air sacs in one or both lungs.', '⚠️'),
    'Bronchiolitis': ('Bronchiolitis', 'Inflammation of the small airways in the lungs.', '🫧'),
}
CLASS_NAMES = ['COPD', 'Healthy', 'URTI', 'Bronchiectasis', 'Pneumonia', 'Bronchiolitis']


# ── SIDEBAR ──
with st.sidebar:
    st.markdown("""
    <div style='padding: 0.5rem 0 1.5rem;'>
        <div style='font-size:1.4rem; font-weight:700; letter-spacing:-0.03em; 
             background:linear-gradient(135deg,#3b82f6,#06b6d4); 
             -webkit-background-clip:text; -webkit-text-fill-color:transparent;
             background-clip:text; margin-bottom:0.25rem;'>RespiAI</div>
        <div style='font-size:0.75rem; color:rgba(100,116,139,0.8); 
             font-weight:500; letter-spacing:0.06em; text-transform:uppercase;'>
             Pulmonary Analysis System</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='sidebar-stat'>🧠 &nbsp;1D CNN Deep Learning</div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-stat'>🎵 &nbsp;193 Audio Features</div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-stat'>🏷️ &nbsp;6 Disease Classes</div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-stat'>📂 &nbsp;WAV · MP3 · M4A</div>", unsafe_allow_html=True)

    st.markdown("<hr style='border:none;border-top:1px solid rgba(148,163,184,0.2);margin:1.25rem 0'>", unsafe_allow_html=True)

    st.markdown("""
    <div style='font-size:0.72rem; font-weight:600; letter-spacing:0.1em; 
         text-transform:uppercase; color:rgba(100,116,139,0.6); margin-bottom:0.6rem;'>
         Feature Extraction</div>
    """, unsafe_allow_html=True)
    for feat, count in [('MFCC Coefficients','40'), ('Chroma Features','12'),
                         ('Mel-spectrogram','128'), ('Spectral Contrast','7'), ('Tonnetz','6')]:
        st.markdown(f"""
        <div style='display:flex;justify-content:space-between;align-items:center;
             padding:0.35rem 0;border-bottom:1px solid rgba(148,163,184,0.12);
             font-size:0.82rem;color:rgba(71,85,105,0.85);'>
            <span>{feat}</span>
            <span style='font-weight:600;color:#3b82f6;'>{count}</span>
        </div>
        """, unsafe_allow_html=True)


# ── HEADER ──
st.markdown("""
<div class='respi-header'>
    <div style='display:flex;align-items:center;gap:0.75rem;margin-bottom:0.75rem;'>
        <span style='font-size:2rem;'>🫁</span>
        <div>
            <h1>Respiratory Disease Prediction</h1>
        </div>
    </div>
    <p>Upload a breathing audio sample and our deep learning model will analyze acoustic 
    patterns to identify potential respiratory conditions with confidence scoring.</p>
    <div style='display:flex;gap:0.6rem;flex-wrap:wrap;margin-top:1rem;'>
        <span class='status-badge badge-blue'>🧠 CNN Model</span>
        <span class='status-badge badge-green'>⚡ Real-time Analysis</span>
        <span class='status-badge badge-amber'>🔬 Research Use Only</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ── DISCLAIMER ──
st.markdown("""
<div class='disclaimer-glass'>
    <span class='disc-icon'>⚠️</span>
    <span><strong>Medical Disclaimer:</strong> This tool is for educational and research purposes only. 
    It is not a substitute for professional medical advice, diagnosis, or treatment. 
    Always consult qualified healthcare providers for medical decisions.</span>
</div>
""", unsafe_allow_html=True)


# ── LOAD MODEL ──
def main():
    model = load_model_safely()

    if model is None:
        st.markdown("""
        <div class='glass-card' style='border-color:rgba(239,68,68,0.3);'>
            <div class='section-title'>❌ Model Loading Error</div>
            <p style='font-size:0.9rem;color:rgba(100,116,139,0.9);'>
            The model file <code>cnn_model.h5</code> could not be loaded. 
            Ensure it exists in the working directory.</p>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    st.markdown("""
    <div style='display:inline-flex;align-items:center;gap:0.5rem;
         background:rgba(16,185,129,0.1);border:1px solid rgba(16,185,129,0.25);
         border-radius:100px;padding:0.4rem 1rem;font-size:0.82rem;
         color:#065f46;font-weight:500;margin-bottom:1.25rem;'>
        ✅ &nbsp;CNN Model loaded — ready for analysis
    </div>
    """, unsafe_allow_html=True)

    # ── UPLOAD SECTION ──
    st.markdown("<div class='section-title'>📤 Upload Audio Sample</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drop your audio file here or click to browse",
        type=['wav', 'mp3', 'm4a'],
        help="Respiratory audio recording — ideally 5+ seconds, quiet environment"
    )

    if uploaded_file is None:
        st.markdown("""
        <div class='glass-card' style='text-align:center;padding:3rem 2rem;'>
            <div style='font-size:3rem;margin-bottom:1rem;opacity:0.6;'>🎙️</div>
            <div style='font-size:1rem;font-weight:500;color:#475569;margin-bottom:0.5rem;'>
                Ready to analyze respiratory sounds</div>
            <div style='font-size:0.85rem;color:rgba(100,116,139,0.7);max-width:360px;margin:0 auto;line-height:1.6;'>
                Upload a WAV, MP3, or M4A recording of breathing sounds. 
                Use a quiet environment and record at least 5 seconds for best results.
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    if not validate_audio_file(uploaded_file):
        st.error("Invalid file format. Please upload WAV, MP3, or M4A.")
        return

    # ── FILE INFO ──
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class='metric-glass'>
            <div class='metric-label'>File Name</div>
            <div class='metric-value' style='font-size:1rem;font-weight:600;word-break:break-all;'>
                {uploaded_file.name}</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class='metric-glass'>
            <div class='metric-label'>File Size</div>
            <div class='metric-value'>{uploaded_file.size/1024:.1f}</div>
            <div class='metric-sub'>kilobytes</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        fmt = uploaded_file.name.split('.')[-1].upper()
        st.markdown(f"""
        <div class='metric-glass'>
            <div class='metric-label'>Format</div>
            <div class='metric-value'>{fmt}</div>
            <div class='metric-sub'>audio format</div>
        </div>""", unsafe_allow_html=True)

    # ── AUDIO PLAYER ──
    st.markdown("<div style='margin:1.25rem 0 0.5rem;' class='section-title'>🔊 Audio Playback</div>",
                unsafe_allow_html=True)
    st.markdown("<div class='glass-card' style='padding:1.25rem;'>", unsafe_allow_html=True)
    st.audio(uploaded_file, format=f'audio/{uploaded_file.name.split(".")[-1]}')
    st.markdown("</div>", unsafe_allow_html=True)

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
            tmp.write(uploaded_file.getbuffer())
            temp_path = tmp.name

        st.markdown("<div class='section-title' style='margin-top:1.25rem;'>🔬 Analysis Results</div>",
                    unsafe_allow_html=True)

        with st.spinner("Extracting acoustic features and running inference…"):
            predicted_class, confidence, all_predictions = predict_single_audio(
                model, temp_path, CLASS_NAMES
            )

        if predicted_class is None:
            st.error("Error processing the audio file. Please try a different recording.")
            return

        is_healthy = predicted_class == "Healthy"
        conf_pct = confidence * 100
        disease_full, disease_desc, disease_icon = disease_info.get(
            predicted_class, (predicted_class, '', '🔍'))

        # ── RESULT LAYOUT ──
        left, right = st.columns([1, 1.1], gap="large")

        with left:
            healthy_class = "healthy" if is_healthy else ""
            bar_class = "healthy" if is_healthy else ""
            st.markdown(f"""
            <div class='prediction-result'>
                <div class='prediction-label'>Primary Detection</div>
                <div class='prediction-disease {healthy_class}'>{disease_icon} {predicted_class}</div>
                <div style='font-size:0.85rem;color:rgba(100,116,139,0.8);margin-bottom:0.75rem;
                     line-height:1.5;'>{disease_full}</div>
                <div style='font-size:0.78rem;color:rgba(100,116,139,0.65);
                     margin-bottom:0.3rem;display:flex;justify-content:space-between;'>
                    <span>Confidence Score</span>
                    <span style='font-weight:600;color:{"#10b981" if is_healthy else "#3b82f6"};'>
                        {conf_pct:.1f}%</span>
                </div>
                <div class='confidence-bar-track'>
                    <div class='confidence-bar-fill {bar_class}' style='width:{conf_pct}%;'></div>
                </div>
                <div style='font-size:0.82rem;color:rgba(100,116,139,0.75);
                     margin-top:0.75rem;line-height:1.55;'>{disease_desc}</div>
            </div>
            """, unsafe_allow_html=True)

            # Recommendation
            if is_healthy:
                st.markdown("""
                <div style='background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.2);
                     border-radius:14px;padding:1rem 1.25rem;margin-top:0.75rem;
                     font-size:0.85rem;color:#065f46;line-height:1.6;'>
                    <strong>✅ Good news!</strong> Analysis suggests normal respiratory sounds.
                    Continue regular health check-ups and maintain good respiratory hygiene.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style='background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.2);
                     border-radius:14px;padding:1rem 1.25rem;margin-top:0.75rem;
                     font-size:0.85rem;color:#78350f;line-height:1.6;'>
                    <strong>⚠️ Potential concern detected:</strong> {predicted_class}<br>
                    Please consult a healthcare professional for proper evaluation. 
                    This is a preliminary analysis only — do not use for self-diagnosis.
                </div>
                """, unsafe_allow_html=True)

            if confidence < 0.6:
                st.markdown("""
                <div style='background:rgba(59,130,246,0.07);border:1px solid rgba(59,130,246,0.18);
                     border-radius:14px;padding:0.85rem 1.25rem;margin-top:0.6rem;
                     font-size:0.82rem;color:#1e40af;line-height:1.5;'>
                    ℹ️ <strong>Low confidence:</strong> Consider re-recording in a quieter 
                    environment for a more reliable result.
                </div>
                """, unsafe_allow_html=True)

        with right:
            # Probability chart
            prob_df = pd.DataFrame({
                'Disease': CLASS_NAMES,
                'Probability': all_predictions * 100
            }).sort_values('Probability', ascending=True)

            colors = ['rgba(59,130,246,0.85)' if d == predicted_class
                      else 'rgba(148,163,184,0.35)' for d in prob_df['Disease']]

            fig = go.Figure(go.Bar(
                x=prob_df['Probability'],
                y=prob_df['Disease'],
                orientation='h',
                marker=dict(
                    color=colors,
                    line=dict(width=0),
                    cornerradius=6,
                ),
                text=[f'{v:.1f}%' for v in prob_df['Probability']],
                textposition='outside',
                textfont=dict(size=11, family='DM Sans'),
            ))
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=40, t=20, b=0),
                height=280,
                xaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(148,163,184,0.15)',
                    zeroline=False,
                    tickfont=dict(size=10, color='rgba(100,116,139,0.7)'),
                    range=[0, max(all_predictions * 100) * 1.2],
                    ticksuffix='%',
                ),
                yaxis=dict(
                    showgrid=False,
                    tickfont=dict(size=11, color='rgba(71,85,105,0.9)', family='DM Sans'),
                ),
                hoverlabel=dict(
                    bgcolor='rgba(15,20,40,0.85)',
                    bordercolor='rgba(255,255,255,0.15)',
                    font=dict(color='white', size=12),
                ),
            )
            st.markdown("<div class='glass-card' style='padding:1.25rem 1.5rem;'>",
                        unsafe_allow_html=True)
            st.markdown("<div class='section-title' style='margin-bottom:0.5rem;'>📊 Confidence Distribution</div>",
                        unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            st.markdown("</div>", unsafe_allow_html=True)

        # ── DETAILED TABLE ──
        st.markdown("<div class='section-title' style='margin-top:1rem;'>📋 Detailed Probability Scores</div>",
                    unsafe_allow_html=True)

        detailed_df = pd.DataFrame({
            'Condition': CLASS_NAMES,
            'Probability': [f'{p*100:.2f}%' for p in all_predictions],
            'Confidence Level': ['High ●' if p > 0.7 else 'Medium ◐' if p > 0.4 else 'Low ○'
                                  for p in all_predictions],
        }).sort_values('Probability', ascending=False,
                       key=lambda x: x.str.rstrip('%').astype(float))

        st.dataframe(detailed_df, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


if __name__ == "__main__":
    main()