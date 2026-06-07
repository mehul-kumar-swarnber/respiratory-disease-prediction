import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
import os

def load_model_safely():
    """Load the pre-trained CNN model safely with error handling"""
    try:
        model_path = "cnn_model.h5"
        if not os.path.exists(model_path):
            st.error(f"Model file not found: {model_path}")
            return None
        
        # Load the model
        model = tf.keras.models.load_model(model_path)
        return model
    
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def validate_audio_file(uploaded_file):
    """Validate if the uploaded file is a valid audio file"""
    try:
        # Check file extension
        valid_extensions = ['.wav', '.mp3', '.m4a']
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        
        if file_extension not in valid_extensions:
            return False
        
        # Check file size (limit to 50MB)
        if uploaded_file.size > 50 * 1024 * 1024:
            st.error("File size too large. Please upload files smaller than 50MB.")
            return False
        
        return True
    
    except Exception:
        return False

def audio_features(filename):
    """Extract audio features using the same method as in training"""
    try:
        sound, sample_rate = librosa.load(filename)
        stft = np.abs(librosa.stft(sound))
        
        # Extract all features (same as training code)
        mfccs = np.mean(librosa.feature.mfcc(y=sound, sr=sample_rate, n_mfcc=40), axis=1)
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate), axis=1)
        mel = np.mean(librosa.feature.melspectrogram(y=sound, sr=sample_rate), axis=1)
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate), axis=1)
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(sound), sr=sample_rate), axis=1)
        
        # Concatenate all features (should result in 193 features)
        concat = np.concatenate((mfccs, chroma, mel, contrast, tonnetz))
        return concat
        
    except Exception as e:
        st.error(f"Error extracting audio features: {str(e)}")
        return None

def predict_single_audio(model, audio_path, class_names):
    """Predict class for a single audio file using CNN model"""
    try:
        # Extract features using the same method as training
        features = audio_features(audio_path)
        
        if features is None:
            return None, None, None
        
        # Reshape features for CNN input (samples, features, channels)
        features = np.reshape(features, (1, features.shape[0], 1))
        
        # Predict using the CNN model
        prediction = model.predict(features, verbose=0)
        
        # Get predicted class and confidence
        predicted_class_idx = np.argmax(prediction)
        confidence = np.max(prediction)
        predicted_class = class_names[predicted_class_idx]
        
        return predicted_class, confidence, prediction[0]
        
    except Exception as e:
        st.error(f"Error predicting audio: {str(e)}")
        return None, None, None

def get_audio_info(audio_path):
    """Get basic information about the audio file"""
    try:
        y, sr = librosa.load(audio_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        
        return {
            'duration': duration,
            'sample_rate': sr,
            'samples': len(y)
        }
    except Exception as e:
        st.error(f"Error getting audio info: {str(e)}")
        return None

def preprocess_audio_for_display(audio_data, sample_rate):
    """Preprocess audio data for visualization"""
    try:
        # Normalize audio
        audio_normalized = audio_data / np.max(np.abs(audio_data))
        
        # Create time axis
        time_axis = np.linspace(0, len(audio_data) / sample_rate, len(audio_data))
        
        return audio_normalized, time_axis
    
    except Exception as e:
        st.error(f"Error preprocessing audio: {str(e)}")
        return None, None

def get_feature_breakdown(features):
    """Break down the 193 features into their components for analysis"""
    try:
        if len(features) != 193:
            return None
        
        breakdown = {
            'MFCC': features[:40],
            'Chroma': features[40:52],
            'Mel-spectrogram': features[52:180],
            'Spectral Contrast': features[180:187],
            'Tonnetz': features[187:193]
        }
        
        return breakdown
        
    except Exception as e:
        st.error(f"Error breaking down features: {str(e)}")
        return None

def display_feature_analysis(features):
    """Display feature analysis in Streamlit"""
    try:
        breakdown = get_feature_breakdown(features)
        
        if breakdown is None:
            return
        
        st.subheader("🔍 Audio Feature Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("MFCC Features", "40")
            st.metric("Chroma Features", "12")
        
        with col2:
            st.metric("Mel-spectrogram", "128")
            st.metric("Spectral Contrast", "7")
        
        with col3:
            st.metric("Tonnetz Features", "6")
            st.metric("Total Features", "193")
        
        # Feature distribution chart
        feature_means = {
            'MFCC': np.mean(breakdown['MFCC']),
            'Chroma': np.mean(breakdown['Chroma']),
            'Mel-spectrogram': np.mean(breakdown['Mel-spectrogram']),
            'Spectral Contrast': np.mean(breakdown['Spectral Contrast']),
            'Tonnetz': np.mean(breakdown['Tonnetz'])
        }
        
        import plotly.express as px
        import pandas as pd
        
        feature_df = pd.DataFrame(
            list(feature_means.items()),
            columns=['Feature Type', 'Mean Value']
        )
        
        fig = px.bar(
            feature_df,
            x='Feature Type',
            y='Mean Value',
            title='Average Feature Values by Type',
            color='Mean Value',
            color_continuous_scale='blues'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error displaying feature analysis: {str(e)}")