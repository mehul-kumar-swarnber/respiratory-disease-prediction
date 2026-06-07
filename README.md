# Respiratory Disease Prediction System

A Streamlit web application that uses deep learning to predict respiratory diseases from audio samples using a pre-trained MobileNetV2 CNN model.

## Features

- **Audio File Upload**: Supports WAV, MP3, and M4A formats
- **Real-time Processing**: Mel spectrogram extraction and analysis
- **Disease Prediction**: Predicts 8 different respiratory conditions
- **Confidence Scoring**: Shows prediction confidence and probability distribution
- **Audio Playback**: Listen to uploaded audio samples
- **Medical Interface**: Professional design with appropriate medical disclaimers

## Supported Conditions

The model can predict the following respiratory conditions:
- Asthma
- Bronchiectasis  
- Bronchiolitis
- COPD (Chronic Obstructive Pulmonary Disease)
- Healthy (Normal)
- LRTI (Lower Respiratory Tract Infection)
- Pneumonia
- URTI (Upper Respiratory Tract Infection)

## Requirements

The application requires the following Python packages:
- streamlit
- tensorflow
- librosa
- opencv-python
- numpy
- plotly
- pandas

## Model Requirements

You need to have the pre-trained model file `lung_sound_mobilenetv2.h5` in the same directory as the application.

## Installation and Usage

1. Ensure you have the required packages installed
2. Place the `lung_sound_mobilenetv2.h5` model file in the project directory
3. Run the application:
   ```bash
   streamlit run app.py --server.port 5000
   ```
4. Open your browser and navigate to the provided URL
5. Upload an audio file and wait for the prediction results

## Audio Guidelines

For best results:
- Use clear, high-quality recordings
- Record in a quiet environment  
- Include at least 5 seconds of respiratory sounds
- Avoid background noise and talking

## Medical Disclaimer

⚠️ **IMPORTANT**: This application is for educational and research purposes only. It is NOT a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers for medical decisions.

## Technical Details

- **Model**: MobileNetV2 CNN architecture
- **Input Processing**: 5-second audio clips converted to mel spectrograms (224x224)
- **Sample Rate**: 22,050 Hz
- **Feature Extraction**: 128 mel bands, 512 hop length, 2048 FFT size
- **Output**: Softmax probabilities for 8 classes

## Architecture

