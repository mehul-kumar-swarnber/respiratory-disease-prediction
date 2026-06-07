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

# Page configuration
st.set_page_config(
    page_title="Respiratory Disease Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Font Awesome and Poppins font, and apply basic font CSS
st.markdown(
    """
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
    html, body, .stApp { font-family: 'Poppins', sans-serif; }
    h1, h2, h3, h4, h5, h6 { font-family: 'Poppins', sans-serif; font-weight:600; }
    .stButton>button, textarea, input, select { font-family: 'Poppins', sans-serif; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Application title and description
st.markdown(
    "<h1 style='margin-bottom:0.15rem'><i class='fas fa-stethoscope' style='color:#2b6cb0;margin-right:8px'></i>Respiratory Disease Prediction System</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    """
    This application uses advanced deep learning to analyze audio samples and predict respiratory conditions.
    Upload an audio recording of breathing sounds to get an AI-powered analysis.
    """
)

# Medical disclaimer
# Expander labels do not accept HTML, so render the icon/title with markdown then use a plain-text expander
st.markdown("<div style='margin-top:6px'><i class='fas fa-exclamation-triangle' style='color:#d69e2e;margin-right:8px'></i><strong>Medical Disclaimer - Please Read</strong></div>", unsafe_allow_html=True)
with st.expander("Read the disclaimer", expanded=False):
    st.warning(
        """
        **IMPORTANT MEDICAL DISCLAIMER:**
        
        This application is for educational and research purposes only. It is NOT a substitute for professional medical advice, diagnosis, or treatment. 
        
        - Always seek the advice of qualified healthcare providers
        - Never disregard professional medical advice based on these results
        - This tool should not be used for medical decision-making
        - Results are predictions based on audio analysis and may not be accurate
        - Consult with a healthcare professional for proper diagnosis and treatment
        """
    )

# Disease information - Updated for CNN model classes
disease_info = {
    'COPD': 'Chronic Obstructive Pulmonary Disease - a group of lung diseases that block airflow.',
    'Healthy': 'Normal respiratory sounds with no abnormalities detected.',
    'URTI': 'Upper Respiratory Tract Infection affecting nose, throat, and airways.',
    'Bronchiectasis': 'A condition where the airways become abnormally widened.',
    'Pneumonia': 'Infection that inflames air sacs in one or both lungs.',
    'Bronchiolitis': 'Inflammation of the small airways in the lungs.'
}

# Class names for the CNN model (matching the order from your training code)
CLASS_NAMES = ['COPD', 'Healthy', 'URTI', 'Bronchiectasis', 'Pneumonia', 'Bronchiolitis']

# Sidebar for model information
with st.sidebar:
    st.markdown("<h3><i class='fas fa-chart-bar' style='color:#2b6cb0;margin-right:8px'></i>Model Information</h3>", unsafe_allow_html=True)
    st.info("""
    **Model:** 1D Convolutional Neural Network (CNN)
    """)

    st.markdown("<h4><i class='fas fa-file-audio' style='color:#38a169;margin-right:8px'></i>Supported Audio Formats</h4>", unsafe_allow_html=True)
    st.write("• WAV files (.wav)")
    st.write("• MP3 files (.mp3)")
    st.write("• M4A files (.m4a)")

    st.markdown("<h4><i class='fas fa-info-circle' style='color:#805ad5;margin-right:8px'></i>How to Use</h4>", unsafe_allow_html=True)
    st.write("1. Upload an audio file")
    st.write("2. Wait for processing")
    st.write("3. Review the prediction results")
    st.write("4. Consult healthcare professionals")

    st.markdown("<h4><i class='fas fa-microscope' style='color:#d53f8c;margin-right:8px'></i>Feature Extraction</h4>", unsafe_allow_html=True)
    st.write("• 40 MFCC coefficients")
    st.write("• 12 Chroma features")
    st.write("• 128 Mel-spectrogram features")
    st.write("• 7 Spectral contrast features")
    st.write("• 6 Tonnetz features")

# Main application
def main():
    # Load the model
    model = load_model_safely()
    
    if model is None:
        st.error("""
        ❌ **Model Loading Error**
        
        The respiratory disease prediction model (cnn_model.h5) could not be loaded.
        Please ensure the model file is present in the current directory.
        """)
        st.stop()
    
    st.markdown(
        "<div style='background:#e6fffa;border-left:4px solid #2f855a;padding:10px;border-radius:6px;display:inline-block'><i class='fas fa-circle-check' style='color:#2f855a;margin-right:8px'></i><strong>CNN Model loaded successfully!</strong></div>",
        unsafe_allow_html=True,
    )
    
    # File upload section
    st.markdown("<h2 style='margin-bottom:0.5rem'><i class='fas fa-upload' style='color:#3182ce;margin-right:8px'></i> Upload Audio Sample</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=['wav', 'mp3', 'm4a'],
        help="Upload a respiratory audio recording (preferably 5 seconds or longer)"
    )
    
    if uploaded_file is not None:
        # Display file information
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Filename:** {uploaded_file.name}")
            st.write(f"**File size:** {uploaded_file.size / 1024:.1f} KB")
        
        # Validate file
        if not validate_audio_file(uploaded_file):
            st.error("❌ Invalid audio file format. Please upload a WAV, MP3, or M4A file.")
            return
        
        # Save uploaded file temporarily
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                temp_path = tmp_file.name
            
            # Audio playback
            st.markdown("<h3 style='margin-bottom:0.25rem'><i class='fas fa-play-circle' style='color:#dd6b20;margin-right:8px'></i> Audio Playback</h3>", unsafe_allow_html=True)
            st.audio(uploaded_file, format=f'audio/{uploaded_file.name.split(".")[-1]}')
            
            # Process and predict
            st.markdown("<h3 style='margin-bottom:0.25rem'><i class='fas fa-vials' style='color:#805ad5;margin-right:8px'></i> Analysis Results</h3>", unsafe_allow_html=True)
            
            with st.spinner("Extracting audio features and generating prediction..."):
                predicted_class, confidence, all_predictions = predict_single_audio(
                    model, temp_path, CLASS_NAMES
                )
                
                if predicted_class is None:
                    st.error("❌ Error processing the audio file. Please try a different file.")
                    return
                
                # Display main prediction
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("<h3 style='margin-bottom:0.25rem'><i class='fas fa-bullseye' style='color:#2b6cb0;margin-right:8px'></i> Primary Prediction</h3>", unsafe_allow_html=True)
                    
                    # Color coding based on condition
                    if predicted_class == "Healthy":
                        st.success(f"**{predicted_class}**")
                    else:
                        st.warning(f"**{predicted_class}**")
                    
                    st.metric("Confidence Score", f"{confidence:.1%}")
                    
                    # Disease information
                    st.info(f"**About {predicted_class}:**\n{disease_info[predicted_class]}")
                
                with col2:
                    st.markdown("<h3 style='margin-bottom:0.25rem'><i class='fas fa-chart-bar' style='color:#2b6cb0;margin-right:8px'></i> Confidence Distribution</h3>", unsafe_allow_html=True)
                    
                    # Create probability chart
                    prob_df = pd.DataFrame({
                        'Disease': CLASS_NAMES,
                        'Probability': all_predictions * 100
                    }).sort_values('Probability', ascending=True)
                    
                    fig = px.bar(
                        prob_df,
                        x='Probability',
                        y='Disease',
                        orientation='h',
                        title='Prediction Probabilities (%)',
                        color='Probability',
                        color_continuous_scale='viridis'
                    )
                    fig.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Detailed results table
                st.markdown("<h3 style='margin-bottom:0.25rem'><i class='fas fa-table' style='color:#4a5568;margin-right:8px'></i> Detailed Probability Scores</h3>", unsafe_allow_html=True)
                
                detailed_df = pd.DataFrame({
                    'Disease': CLASS_NAMES,
                    'Probability (%)': [f"{prob*100:.2f}%" for prob in all_predictions],
                    'Confidence': ['High' if prob > 0.7 else 'Medium' if prob > 0.4 else 'Low' for prob in all_predictions]
                }).sort_values('Probability (%)', ascending=False, key=lambda x: x.str.rstrip('%').astype(float))
                
                st.dataframe(detailed_df, use_container_width=True)
                
                # Recommendations
                st.markdown("<h3 style='margin-bottom:0.25rem'><i class='fas fa-lightbulb' style='color:#ecc94b;margin-right:8px'></i> Recommendations</h3>", unsafe_allow_html=True)
                
                if predicted_class == "Healthy":
                    st.success("""
                    ✅ **Good News!** The analysis suggests normal respiratory sounds.
                    
                    **Next Steps:**
                    - Continue maintaining good respiratory health
                    - Regular health check-ups as recommended by your doctor
                    - If you have symptoms, consult a healthcare professional
                    """)
                else:
                    st.warning(f"""
                    ⚠️ **Potential Concern Detected:** {predicted_class}
                    
                    **Important Next Steps:**
                    - Consult with a healthcare professional immediately
                    - This is a preliminary analysis and requires medical confirmation
                    - Do not use this result for self-diagnosis or treatment decisions
                    - Seek professional medical evaluation and testing
                    """)
                
                # Additional insights
                if confidence < 0.6:
                    st.markdown(
                        "<div style='background:#ebf8ff;border-left:4px solid #3182ce;padding:10px;border-radius:6px'><i class='fas fa-circle-info' style='color:#3182ce;margin-right:8px'></i><strong>Low Confidence Alert</strong><div style='margin-top:6px'>The model's confidence is relatively low. This could indicate:<ul><li>Audio quality issues</li><li>Ambiguous respiratory sounds</li><li>Need for additional analysis</li></ul>Consider recording in a quieter environment or consulting a healthcare professional for definitive diagnosis.</div></div>",
                        unsafe_allow_html=True,
                    )
                
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
        
        finally:
            # Clean up temporary file
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
    
    else:
        # Instructions when no file is uploaded
        st.markdown("""
        <div style='background:#f7fafc;border-left:4px solid #4a5568;padding:12px;border-radius:6px'>
        <i class='fas fa-hand-pointer' style='margin-right:6px'></i> <strong>Ready to analyze respiratory sounds!</strong>
        <div style='margin-top:8px'>Please upload an audio file to begin the analysis. For best results:
        <ul>
        <li>Use clear, high-quality recordings</li>
        <li>Record in a quiet environment</li>
        <li>Include at least 5 seconds of respiratory sounds</li>
        <li>Common formats: WAV, MP3, M4A</li>
        </ul>
        </div>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()