import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import plotly.express as px
import pandas as pd
import os
import tempfile
from utils import predict_single_audio, validate_audio_file, load_model_safely

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Respiratory Disease Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# Font Awesome 6 (CDN) handles all custom icons — no install required.
# Typography is scoped to custom classes only so Streamlit's own widget
# icons (sidebar toggle, expander arrow) keep their Material Symbols font.
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<link rel="stylesheet"
  href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap"
  rel="stylesheet">

<style>
/* ── 1. FORCE LIGHT THEME on every surface ─────────────────────────────── */
html, body { color-scheme: light !important; background:#f5f7fa !important; }

.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main,
.block-container { background-color:#f5f7fa !important; color:#1a202c !important; }

[data-testid="stHeader"] {
    height: 2.5rem;
},
[data-testid="stToolbar"],
[data-testid="stDecoration"] { background-color:#f5f7fa !important; }

section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div,
section[data-testid="stSidebar"] > div > div {
  background-color:#ffffff !important;
  border-right:1px solid #e2e8f0 !important;
}

/* ── 2. TYPOGRAPHY ─────────────────────────────────────────────────────── */
/* Only target our own classes + safe native elements — never span/div globally
   because that overwrites Streamlit's Material Symbols font on its own widgets */
html, body, p, li, td, th, label, input, textarea, select,
.stMarkdown, .stMarkdown p, .stMarkdown li,
.info-card, .section-lbl, .tag-pill, .proc-step,
.loader-title, .loader-sub,
.pg-h1, .pg-h2, .pg-h3, .pg-h4 {
  font-family: 'Poppins', sans-serif !important;
  color: #1a202c !important;
}

h1, h2, h3, h4, h5, h6 {
  font-family: 'DM Serif Display', serif !important;
  color: #1a202c !important;
}

/* ── 3. FA ICON — never inherit the serif/Poppins family ───────────────── */
i.fas, i.far, i.fab, i.fa, i.fa-solid, i.fa-regular, i.fa-brands,
[class*="fa-"] {
  font-family: "Font Awesome 6 Free", "Font Awesome 6 Brands" !important;
  font-style: normal !important;
  color: inherit;
}



/* ── 5. FILE UPLOADER ───────────────────────────────────────────────────── */
[data-testid="stFileUploader"],
[data-testid="stFileUploader"] section,
[data-testid="stFileUploader"] section > div,
[data-testid="stFileUploader"] section > input,
[data-testid="stFileUploaderDropzone"],
[data-testid="stFileUploaderDropzoneInstructions"] {
  background-color:#ffffff !important;
  color: #1a202c !important;
}
[data-testid="stFileUploader"] {
  border: 2px dashed #bee3f8 !important;
  border-radius: 12px !important;
  padding: 1.5rem !important;
}
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] p {
  color: #1a202c !important;
  background-color: transparent !important;
}
[data-testid="stFileUploaderDropzone"] button {
    background: linear-gradient(135deg, #3182ce, #4299e1) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;

    padding: 0.65rem 1.25rem !important;
    font-weight: 600 !important;

    box-shadow: 0 4px 12px rgba(49, 130, 206, 0.25) !important;

    transition: all 0.2s ease !important;
}

[data-testid="stFileUploaderDropzone"] button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(49, 130, 206, 0.35) !important;
}

/* ── 6. BUTTONS ─────────────────────────────────────────────────────────── */
.stButton > button {
  background:#2b6cb0 !important;
  color:#ffffff !important;
  border:none !important;
  border-radius:8px !important;
  font-weight:600 !important;
  transition:background .2s !important;
}
.stButton > button:hover { background:#2c5282 !important; }

/* ── 7. ALERTS / DATAFRAME ──────────────────────────────────────────────── */
[data-testid="stAlert"] { color:#1a202c !important; }
div[data-baseweb="notification"] { background:#f0fff4 !important; }
[data-testid="stDataFrame"] { background:#ffffff !important; }

/* ── 8. AUDIO LOADER ────────────────────────────────────────────────────── */
.audio-loader {
  display:flex; flex-direction:column; align-items:center;
  gap:1.4rem; padding:2.75rem 1rem;
  background:#ffffff; border:1px solid #e2e8f0;
  border-radius:16px; margin:.5rem 0;
}
.waveform { display:flex; align-items:center; gap:5px; height:56px; }
.waveform span {
  display:block; width:5px; border-radius:99px; background:#2b6cb0;
  animation:waveBounce 1.1s ease-in-out infinite;
}
.waveform span:nth-child(1)  {height:18px;animation-delay:.00s}
.waveform span:nth-child(2)  {height:34px;animation-delay:.09s}
.waveform span:nth-child(3)  {height:46px;animation-delay:.18s}
.waveform span:nth-child(4)  {height:56px;animation-delay:.27s}
.waveform span:nth-child(5)  {height:40px;animation-delay:.36s}
.waveform span:nth-child(6)  {height:56px;animation-delay:.45s}
.waveform span:nth-child(7)  {height:30px;animation-delay:.54s}
.waveform span:nth-child(8)  {height:46px;animation-delay:.63s}
.waveform span:nth-child(9)  {height:22px;animation-delay:.72s}
.waveform span:nth-child(10) {height:38px;animation-delay:.81s}
.waveform span:nth-child(11) {height:52px;animation-delay:.90s}
.waveform span:nth-child(12) {height:28px;animation-delay:.14s}
@keyframes waveBounce {
  0%,100%{transform:scaleY(.35);opacity:.45}
  50%    {transform:scaleY(1);opacity:1}
}
.loader-title {
  font-family:'DM Serif Display',serif !important;
  font-size:1.2rem; color:#2b6cb0 !important;
}
.loader-sub {
  font-size:.78rem; color:#718096 !important;
  text-transform:uppercase; letter-spacing:.06em; margin-top:-.8rem;
}
.proc-steps{display:flex;flex-direction:column;gap:.45rem;width:100%;max-width:340px}
.proc-step {
  display:flex; align-items:center; gap:.65rem; font-size:.82rem;
  color:#4a5568 !important; background:#f7fafc;
  border:1px solid #e2e8f0; border-radius:9px; padding:.5rem .9rem;
  opacity:0; animation:stepIn .4s ease forwards;
}
.proc-step:nth-child(1){animation-delay:.15s}
.proc-step:nth-child(2){animation-delay:.75s}
.proc-step:nth-child(3){animation-delay:1.35s}
.proc-step:nth-child(4){animation-delay:1.95s}
@keyframes stepIn{to{opacity:1}}
.proc-step i { width:16px; text-align:center; flex-shrink:0; }

/* ── 9. CARD / UTILITY ──────────────────────────────────────────────────── */
.info-card {
  background:#ffffff; border:1px solid #e2e8f0;
  border-radius:14px; padding:1.25rem 1.5rem; margin-bottom:1rem;
}
.section-lbl {
  font-size:.68rem; font-weight:600; letter-spacing:.1em;
  text-transform:uppercase; color:#718096 !important; margin-bottom:.3rem;
  margin-top:.6rem;
}
.tag-pill {
  display:inline-block; background:#ebf8ff; color:#2b6cb0 !important;
  border-radius:99px; padding:.2rem .8rem; font-size:.75rem;
  font-weight:600; margin:.15rem;
}

/* ── 10. SECTION HEADINGS (plain div, not h-tags) ───────────────────────── */
.pg-h1 {
  font-family:'DM Serif Display',serif; font-size:2rem; font-weight:400;
  color:#1a202c; margin-bottom:.15rem; display:flex; align-items:center; gap:.5rem;
}
.pg-h2 {
  font-family:'DM Serif Display',serif; font-size:1.5rem; font-weight:400;
  color:#1a202c; margin-bottom:.5rem; display:flex; align-items:center; gap:.5rem;
}
.pg-h3 {
  font-family:'DM Serif Display',serif; font-size:1.15rem; font-weight:400;
  color:#1a202c; margin-bottom:.35rem; display:flex; align-items:center; gap:.45rem;
}
.pg-h4 {
  font-family:'Poppins',sans-serif; font-size:.95rem; font-weight:600;
  color:#1a202c; margin-bottom:.4rem; display:flex; align-items:center; gap:.4rem;
}
.pg-h1 i,.pg-h2 i,.pg-h3 i,.pg-h4 i {
  font-size:1rem !important;
  font-family:"Font Awesome 6 Free" !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# POPUP  —  via components.html (scripts only run here in Streamlit)
# ─────────────────────────────────────────────────────────────────────────────
components.html("""
<!DOCTYPE html><html><head>
<link rel="stylesheet"
  href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&family=DM+Serif+Display&display=swap"
  rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:transparent;font-family:'Poppins',sans-serif}
#ov{
  position:fixed;inset:0;
  background:rgba(10,15,25,.65);
  backdrop-filter:blur(6px);
  display:flex;align-items:center;justify-content:center;
  z-index:2147483647;
  transition:opacity .35s ease;
}
#ov.gone{opacity:0;pointer-events:none}
#card{
  background:#fff;border-radius:20px;padding:2.5rem 2.75rem;
  max-width:450px;width:92%;
  box-shadow:0 40px 80px rgba(0,0,0,.3);
  text-align:center;
  animation:up .45s cubic-bezier(.16,1,.3,1) both;
}
@keyframes up{from{opacity:0;transform:translateY(28px) scale(.97)}to{opacity:1;transform:none}}
.ico{
  width:66px;height:66px;background:#fff7ed;border-radius:50%;
  display:flex;align-items:center;justify-content:center;
  margin:0 auto 1.2rem;font-size:1.8rem;color:#dd6b20;
}
h2{font-family:'DM Serif Display',serif;font-size:1.35rem;color:#1a202c;margin-bottom:.7rem}
p{font-size:.875rem;color:#4a5568;line-height:1.75;margin-bottom:1.35rem}
.bar{height:5px;background:#e2e8f0;border-radius:99px;overflow:hidden;margin-bottom:1.3rem}
.fill{
  height:100%;background:linear-gradient(90deg,#2b6cb0,#63b3ed);
  border-radius:99px;transform-origin:left;
  animation:sh 5s linear forwards;
}
@keyframes sh{from{transform:scaleX(1)}to{transform:scaleX(0)}}
button{
  display:inline-flex;align-items:center;gap:.45rem;
  background:#2b6cb0;color:#fff;border:none;border-radius:9px;
  padding:.6rem 1.6rem;font-family:'Poppins',sans-serif;
  font-size:.875rem;font-weight:600;cursor:pointer;transition:background .2s;
}
button:hover{background:#2c5282}
</style></head><body>
<div id="ov">
  <div id="card">
    <div class="ico"><i class="fas fa-triangle-exclamation"></i></div>
    <h2>Educational Purpose Only</h2>
    <p>
      This application is for <strong>educational &amp; research use only</strong>.<br>
      It is <em>not clinically proven</em> and must not be used for real diagnosis,
      treatment, or clinical decisions.<br><br>
      Always consult a qualified healthcare professional.
    </p>
    <div class="bar"><div class="fill"></div></div>
    <button onclick="close_()">
      <i class="fas fa-circle-check"></i>&nbsp;I Understand
    </button>
  </div>
</div>
<script>
var KEY='rdp_edu_v2';
function close_(){
  try{window.parent.sessionStorage.setItem(KEY,'1')}catch(e){}
  try{sessionStorage.setItem(KEY,'1')}catch(e){}
  var o=document.getElementById('ov');
  o.classList.add('gone');
  setTimeout(function(){o.style.display='none'},370);
}
var seen=false;
try{seen=window.parent.sessionStorage.getItem(KEY)==='1'}catch(e){}
if(!seen){try{seen=sessionStorage.getItem(KEY)==='1'}catch(e){}}
if(seen){document.getElementById('ov').style.display='none'}
else{setTimeout(close_,5000)}
</script>
</body></html>
""", height=0, scrolling=False)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
disease_info = {
    'COPD':           'Chronic Obstructive Pulmonary Disease — blocks airflow and makes breathing progressively harder.',
    'Healthy':        'Normal respiratory sounds with no abnormalities detected.',
    'URTI':           'Upper Respiratory Tract Infection — affects the nose, throat, and upper airways.',
    'Bronchiectasis': 'Airways become abnormally widened and cannot clear mucus effectively.',
    'Pneumonia':      'Infection inflaming air sacs in one or both lungs, which may fill with fluid.',
    'Bronchiolitis':  'Inflammation of the small airways (bronchioles) in the lungs.',
}
CLASS_NAMES = ['COPD', 'Healthy', 'URTI', 'Bronchiectasis', 'Pneumonia', 'Bronchiolitis']

# helper: render a heading as a div (avoids h-tag + FA icon font conflicts)
def H(level, icon_cls, text, color="#2b6cb0"):
    return (f"<div class='pg-h{level}'>"
            f"<i class='fas {icon_cls}' style='color:{color}'></i>"
            f"{text}</div>")

def section_label(txt, icon=""):
    ico = f"<i class='fas {icon}' style='margin-right:5px'></i>" if icon else ""
    return f"<div class='section-lbl'>{ico}{txt}</div>"


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(H(3, "fa-brain", "Model Information"), unsafe_allow_html=True)
    st.markdown(
        "<div class='info-card'>"
        + section_label("Architecture", "fa-microchip")
        + "<span class='tag-pill'><i class='fas fa-network-wired' style='margin-right:5px'></i>1D CNN</span>"
        + section_label("Framework", "fa-layer-group")
        + "<span class='tag-pill'><i class='fas fa-cubes' style='margin-right:5px'></i>TensorFlow / Keras</span>"
        + "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(H(4, "fa-file-audio", "Supported Formats", "#38a169"), unsafe_allow_html=True)
    for fmt, ext in [("WAV", ".wav"), ("MP3", ".mp3"), ("M4A", ".m4a")]:
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:5px'>"
            f"<i class='fas fa-circle' style='color:#2b6cb0;font-size:.4rem'></i>"
            f"<span><strong>{fmt}</strong> &mdash; {ext}</span></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-top:1rem'>" + H(4, "fa-list-ol", "How to Use", "#805ad5") + "</div>",
                unsafe_allow_html=True)
    for n, (ico, txt) in enumerate([
        ("fa-upload",      "Upload an audio file"),
        ("fa-gear",        "Wait for feature extraction"),
        ("fa-chart-pie",   "Review prediction results"),
        ("fa-user-doctor", "Consult healthcare professionals"),
    ], 1):
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:7px'>"
            f"<span style='background:#ebf8ff;color:#2b6cb0;border-radius:50%;width:22px;height:22px;"
            f"display:inline-flex;align-items:center;justify-content:center;"
            f"font-size:.7rem;font-weight:700;flex-shrink:0'>{n}</span>"
            f"<i class='fas {ico}' style='color:#2b6cb0;width:14px;text-align:center'></i>"
            f"<span style='font-size:.85rem'>{txt}</span></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-top:1rem'>" + H(4, "fa-microscope", "Feature Extraction", "#d53f8c") + "</div>",
                unsafe_allow_html=True)
    for ico, lbl in [
        ("fa-signal",     "40 MFCC coefficients"),
        ("fa-music",      "12 Chroma features"),
        ("fa-chart-area", "128 Mel-spectrogram features"),
        ("fa-sliders",    "7 Spectral contrast features"),
        ("fa-circle-dot", "6 Tonnetz features"),
    ]:
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:5px'>"
            f"<i class='fas {ico}' style='color:#d53f8c;width:14px;text-align:center'></i>"
            f"<span style='font-size:.83rem'>{lbl}</span></div>",
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# AUDIO LOADER
# ─────────────────────────────────────────────────────────────────────────────
def audio_loader_html():
    bars = "".join("<span></span>" for _ in range(12))
    return f"""
    <div class='audio-loader'>
      <div class='waveform'>{bars}</div>
      <div class='loader-title'>
        <i class='fas fa-volume-high' style='margin-right:8px;color:#63b3ed'></i>
        Processing Audio
      </div>
      <div class='loader-sub'>
        <i class='fas fa-spinner fa-spin' style='margin-right:6px'></i>
        Extracting features &amp; running inference
      </div>
      <div class='proc-steps'>
        <div class='proc-step'>
          <i class='fas fa-file-audio' style='color:#2b6cb0'></i>
          <span>Loading &amp; resampling audio signal</span>
        </div>
        <div class='proc-step'>
          <i class='fas fa-signal' style='color:#2b6cb0'></i>
          <span>Computing MFCC &amp; chroma features</span>
        </div>
        <div class='proc-step'>
          <i class='fas fa-chart-area' style='color:#2b6cb0'></i>
          <span>Building mel-spectrogram representation</span>
        </div>
        <div class='proc-step'>
          <i class='fas fa-brain' style='color:#2b6cb0'></i>
          <span>Running CNN inference model</span>
        </div>
      </div>
    </div>"""


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # Page header
    st.markdown(H(1, "fa-stethoscope", "Respiratory Disease Prediction System"), unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#4a5568;margin-bottom:1.5rem;font-size:.93rem'>"
        "<i class='fas fa-circle-info' style='margin-right:6px;color:#63b3ed'></i>"
        "Advanced deep learning analysis of breathing audio samples to predict respiratory conditions.</p>",
        unsafe_allow_html=True,
    )

    # Disclaimer expander
    st.markdown(
        "<div style='margin-bottom:.25rem'>"
        "<i class='fas fa-triangle-exclamation' style='color:#d69e2e;margin-right:8px'></i>"
        "<strong>Medical Disclaimer</strong></div>",
        unsafe_allow_html=True,
    )
    with st.expander("Read the full disclaimer", expanded=False):
        st.markdown(
            "<ul style='line-height:2;padding-left:1.1rem;margin:0'>"
            "<li><i class='fas fa-xmark' style='color:#c53030;margin-right:6px'></i>"
            "For <strong>educational and research purposes only</strong>.</li>"
            "<li><i class='fas fa-xmark' style='color:#c53030;margin-right:6px'></i>"
            "Not a substitute for professional medical advice or diagnosis.</li>"
            "<li><i class='fas fa-xmark' style='color:#c53030;margin-right:6px'></i>"
            "Predictions may not be accurate — clinical confirmation required.</li>"
            "<li><i class='fas fa-check' style='color:#38a169;margin-right:6px'></i>"
            "Always consult a qualified healthcare professional.</li>"
            "</ul>",
            unsafe_allow_html=True,
        )

    # Load model
    model = load_model_safely()
    if model is None:
        st.markdown(
            "<div class='info-card' style='border-color:#fed7d7'>"
            "<i class='fas fa-circle-xmark' style='color:#c53030;margin-right:8px'></i>"
            "<strong>Model Loading Error</strong>"
            "<p style='margin-top:.5rem;font-size:.875rem'>"
            "The file <code>cnn_model.h5</code> could not be found. "
            "Ensure it is present in the current directory and restart the app.</p></div>",
            unsafe_allow_html=True,
        )
        st.stop()

    st.markdown(
        "<div style='background:#f0fff4;border-left:4px solid #38a169;padding:10px 14px;"
        "border-radius:8px;display:inline-flex;align-items:center;gap:8px;margin-bottom:1.5rem'>"
        "<i class='fas fa-circle-check' style='color:#38a169'></i>"
        "<strong>CNN model loaded successfully</strong></div>",
        unsafe_allow_html=True,
    )

    # Upload section
    st.markdown(H(2, "fa-cloud-arrow-up", "Upload Audio Sample", "#3182ce"), unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose an audio file (WAV · MP3 · M4A — minimum 5 seconds recommended)",
        type=["wav", "mp3", "m4a"],
        help="Upload a respiratory audio recording for AI-powered analysis",
    )

    if uploaded_file is not None:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(
                f"<div class='info-card'>"
                f"{section_label('File name', 'fa-file')}"
                f"<strong>{uploaded_file.name}</strong>"
                f"{section_label('Size', 'fa-weight-hanging')}"
                f"<strong>{uploaded_file.size/1024:.1f} KB</strong>"
                f"</div>",
                unsafe_allow_html=True,
            )

        if not validate_audio_file(uploaded_file):
            st.markdown(
                "<div class='info-card' style='border-color:#fed7d7'>"
                "<i class='fas fa-circle-xmark' style='color:#c53030;margin-right:8px'></i>"
                "Invalid audio format. Please upload a WAV, MP3, or M4A file.</div>",
                unsafe_allow_html=True,
            )
            return

        temp_path = None
        try:
            suffix = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getbuffer())
                temp_path = tmp.name

            st.markdown(H(3, "fa-circle-play", "Audio Playback", "#dd6b20"), unsafe_allow_html=True)
            st.audio(uploaded_file, format=f"audio/{suffix.lstrip('.')}")

            st.markdown(
                "<div style='margin-top:1.25rem'>" +
                H(3, "fa-flask", "Analysis Results", "#805ad5") +
                "</div>",
                unsafe_allow_html=True,
            )

            loader_slot = st.empty()
            loader_slot.markdown(audio_loader_html(), unsafe_allow_html=True)

            predicted_class, confidence, all_predictions = predict_single_audio(
                model, temp_path, CLASS_NAMES
            )
            loader_slot.empty()

            if predicted_class is None:
                st.markdown(
                    "<div class='info-card' style='border-color:#fed7d7'>"
                    "<i class='fas fa-circle-xmark' style='color:#c53030;margin-right:8px'></i>"
                    "Error processing the audio file. Please try a different recording.</div>",
                    unsafe_allow_html=True,
                )
                return

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(H(4, "fa-bullseye", "Primary Prediction"), unsafe_allow_html=True)
                accent = "#38a169" if predicted_class == "Healthy" else "#c05621"
                bg     = "#f0fff4" if predicted_class == "Healthy" else "#fffaf0"
                ico    = "fa-circle-check" if predicted_class == "Healthy" else "fa-triangle-exclamation"
                st.markdown(
                    f"<div class='info-card' style='border-color:{accent};background:{bg}'>"
                    f"<div style='display:flex;align-items:center;gap:10px;margin-bottom:.75rem'>"
                    f"<i class='fas {ico}' style='color:{accent};font-size:1.4rem'></i>"
                    f"<span style='font-size:1.3rem;font-weight:700;color:{accent}'>{predicted_class}</span>"
                    f"</div>"
                    f"{section_label('Confidence Score')}"
                    f"<div style='font-size:1.55rem;font-weight:700;color:{accent}'>{confidence:.1%}</div>"
                    f"{section_label('About this condition')}"
                    f"<p style='font-size:.82rem;margin:0;color:#4a5568'>{disease_info[predicted_class]}</p>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            with col2:
                st.markdown(H(4, "fa-chart-bar", "Confidence Distribution"), unsafe_allow_html=True)
                prob_df = pd.DataFrame({
                    "Disease":     CLASS_NAMES,
                    "Probability": all_predictions * 100,
                }).sort_values("Probability", ascending=True)
                fig = px.bar(
                    prob_df, x="Probability", y="Disease", orientation="h",
                    title="Prediction Probabilities (%)", color="Probability",
                    color_continuous_scale=[[0,"#bee3f8"],[1,"#2b6cb0"]],
                )
                fig.update_layout(
                    height=360, showlegend=False,
                    paper_bgcolor="#ffffff", plot_bgcolor="#f7fafc",
                    font=dict(family="Poppins, sans-serif", color="#1a202c"),
                    title_font=dict(family="DM Serif Display, serif"),
                    margin=dict(l=0,r=10,t=40,b=10),
                )
                fig.update_coloraxes(showscale=False)
                fig.update_xaxes(showgrid=True, gridcolor="#e2e8f0", zeroline=False,
                                 tickfont=dict(color="#4a5568"))
                fig.update_yaxes(tickfont=dict(color="#4a5568"))
                st.plotly_chart(fig, use_container_width=True)

            st.markdown(H(4, "fa-table", "Detailed Probability Scores", "#4a5568"), unsafe_allow_html=True)
            detailed_df = pd.DataFrame({
                "Condition":       CLASS_NAMES,
                "Probability (%)": [f"{p*100:.2f}%" for p in all_predictions],
                "Confidence":      ["High" if p>.7 else "Medium" if p>.4 else "Low"
                                    for p in all_predictions],
            }).sort_values("Probability (%)", ascending=False,
                           key=lambda x: x.str.rstrip("%").astype(float))
            st.dataframe(detailed_df, use_container_width=True, hide_index=True)

            st.markdown(H(4, "fa-lightbulb", "Recommendations", "#d69e2e"), unsafe_allow_html=True)
            if predicted_class == "Healthy":
                st.markdown(
                    "<div class='info-card' style='border-color:#38a169;background:#f0fff4'>"
                    "<p style='margin:0 0 .5rem'>"
                    "<i class='fas fa-circle-check' style='color:#38a169;margin-right:8px'></i>"
                    "<strong>Good News — Normal Respiratory Sounds Detected</strong></p>"
                    "<ul style='margin:0;padding-left:1.25rem;line-height:2'>"
                    "<li><i class='fas fa-arrow-right' style='color:#38a169;margin-right:6px'></i>"
                    "Continue maintaining good respiratory health habits.</li>"
                    "<li><i class='fas fa-arrow-right' style='color:#38a169;margin-right:6px'></i>"
                    "Schedule regular health check-ups as advised by your doctor.</li>"
                    "<li><i class='fas fa-arrow-right' style='color:#38a169;margin-right:6px'></i>"
                    "If symptoms arise, consult a healthcare professional promptly.</li>"
                    "</ul></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div class='info-card' style='border-color:#c05621;background:#fffaf0'>"
                    f"<p style='margin:0 0 .5rem'>"
                    f"<i class='fas fa-triangle-exclamation' style='color:#c05621;margin-right:8px'></i>"
                    f"<strong>Potential Concern Detected — {predicted_class}</strong></p>"
                    f"<ul style='margin:0;padding-left:1.25rem;line-height:2'>"
                    f"<li><i class='fas fa-user-doctor' style='color:#c05621;margin-right:6px'></i>"
                    f"Consult a healthcare professional at the earliest opportunity.</li>"
                    f"<li><i class='fas fa-vials' style='color:#c05621;margin-right:6px'></i>"
                    f"This is a preliminary analysis — clinical confirmation is required.</li>"
                    f"<li><i class='fas fa-ban' style='color:#c05621;margin-right:6px'></i>"
                    f"Do not self-diagnose or make treatment decisions from this result.</li>"
                    f"<li><i class='fas fa-hospital' style='color:#c05621;margin-right:6px'></i>"
                    f"Seek professional medical evaluation and appropriate testing.</li>"
                    f"</ul></div>",
                    unsafe_allow_html=True,
                )

            if confidence < 0.6:
                st.markdown(
                    "<div class='info-card' style='border-color:#3182ce;background:#ebf8ff'>"
                    "<p style='margin:0 0 .4rem'>"
                    "<i class='fas fa-circle-info' style='color:#3182ce;margin-right:8px'></i>"
                    "<strong>Low Confidence Notice</strong></p>"
                    "<p style='margin:0;font-size:.83rem;color:#4a5568'>"
                    "The model confidence is relatively low — possibly due to audio quality "
                    "or background noise. Try re-recording in a quieter environment or consult "
                    "a healthcare professional for a definitive diagnosis.</p></div>",
                    unsafe_allow_html=True,
                )

        except Exception as exc:
            st.markdown(
                f"<div class='info-card' style='border-color:#fed7d7'>"
                f"<i class='fas fa-circle-xmark' style='color:#c53030;margin-right:8px'></i>"
                f"<strong>Error processing file:</strong> {exc}</div>",
                unsafe_allow_html=True,
            )
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)

    else:
        st.markdown(
            "<div class='info-card' style='text-align:center;padding:2.75rem 1.5rem'>"
            "<i class='fas fa-cloud-arrow-up' style='font-size:2.5rem;color:#bee3f8;"
            "margin-bottom:1rem;display:block'></i>"
            "<div class='pg-h3' style='justify-content:center;margin-bottom:.5rem'>"
            "Ready to Analyse Respiratory Sounds</div>"
            "<p style='color:#718096;font-size:.875rem;max-width:380px;margin:0 auto 1.25rem'>"
            "Upload an audio file to begin the AI-powered analysis. For best results, "
            "use a clear recording in a quiet environment.</p>"
            "<div style='display:flex;flex-wrap:wrap;justify-content:center;gap:.5rem'>"
            "<span class='tag-pill'><i class='fas fa-microphone' style='margin-right:5px'></i>5 sec minimum</span>"
            "<span class='tag-pill'><i class='fas fa-volume-high' style='margin-right:5px'></i>Clear audio</span>"
            "<span class='tag-pill'><i class='fas fa-wind' style='margin-right:5px'></i>Quiet environment</span>"
            "<span class='tag-pill'><i class='fas fa-file-audio' style='margin-right:5px'></i>WAV / MP3 / M4A</span>"
            "</div></div>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()