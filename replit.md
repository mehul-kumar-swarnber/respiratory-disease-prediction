# Respiratory Disease Prediction System

## Overview

This is a Streamlit web application that uses deep learning to predict respiratory diseases from audio samples. The system employs a pre-trained MobileNetV2 CNN model to analyze mel spectrograms extracted from audio recordings and classify them into 8 different respiratory conditions including healthy, asthma, COPD, pneumonia, and various respiratory tract infections.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web application framework
- **Layout**: Wide layout with expandable sidebar
- **Components**: File uploader, audio player, prediction results display, interactive charts
- **Visualization**: Plotly for interactive graphs and confidence score visualization

### Backend Architecture
- **ML Framework**: TensorFlow/Keras for deep learning model inference
- **Audio Processing**: Librosa for audio file loading and feature extraction
- **Image Processing**: OpenCV for mel spectrogram manipulation
- **Model**: Pre-trained MobileNetV2 CNN model stored as `lung_sound_mobilenetv2.h5`

### Data Processing Pipeline
1. Audio file upload and validation
2. Audio loading with 5-second duration limit at 22,050 Hz sample rate
3. Mel spectrogram extraction and conversion to image format
4. Image preprocessing for CNN input (224x224 pixels)
5. Model inference and probability calculation
6. Results visualization and display

## Key Components

### Core Modules
- **app.py**: Main Streamlit application with UI components and user interaction
- **utils.py**: Utility functions for model loading, audio validation, and prediction processing
- **lung_sound_mobilenetv2.h5**: Pre-trained deep learning model file (required dependency)

### Disease Classification
The system predicts 8 respiratory conditions:
- Healthy (Normal breathing)
- Asthma
- Bronchiectasis
- Bronchiolitis
- COPD (Chronic Obstructive Pulmonary Disease)
- LRTI (Lower Respiratory Tract Infection)
- Pneumonia
- URTI (Upper Respiratory Tract Infection)

### Audio Processing Features
- **Supported Formats**: WAV, MP3, M4A
- **File Size Limit**: 50MB maximum
- **Duration**: Processes 5-second audio segments
- **Sample Rate**: 22,050 Hz standardization
- **Feature Extraction**: Mel spectrogram conversion to visual representation

## Data Flow

1. **Input Stage**: User uploads audio file through Streamlit file uploader
2. **Validation**: File format and size validation using utility functions
3. **Processing**: Audio loading, mel spectrogram extraction, and image preprocessing
4. **Prediction**: CNN model inference on processed spectrogram
5. **Output**: Confidence scores, probability distribution, and disease information display
6. **Visualization**: Interactive charts showing prediction confidence and probability breakdown

## External Dependencies

### Python Packages
- **streamlit**: Web application framework
- **tensorflow**: Deep learning model inference
- **librosa**: Audio file processing and feature extraction
- **opencv-python**: Image processing and manipulation
- **numpy**: Numerical computations
- **plotly**: Interactive data visualization
- **pandas**: Data manipulation and analysis

### Model Dependencies
- Pre-trained MobileNetV2 model file (`lung_sound_mobilenetv2.h5`) must be present in the project directory
- Model expects 224x224 pixel input images (mel spectrograms)

## Deployment Strategy

### Local Development
- Run using `streamlit run app.py --server.port 5000`
- Requires all dependencies installed in Python environment
- Model file must be accessible in the same directory

### Production Considerations
- **Medical Compliance**: Includes comprehensive medical disclaimers
- **Error Handling**: Robust error handling for model loading and audio processing
- **File Validation**: Input validation for audio files and size limits
- **Performance**: Optimized for real-time processing of audio samples

### Security and Legal
- Medical disclaimer prominently displayed to users
- Clear warnings about educational/research purpose only
- File size limitations to prevent resource abuse
- Input validation to prevent malicious file uploads

The application is designed as a demonstration/research tool and includes appropriate medical disclaimers emphasizing that it should not be used for actual medical diagnosis or treatment decisions.