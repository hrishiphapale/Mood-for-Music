import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
from PIL import Image
import pandas as pd
import os

# Define model paths
MOOD_CLASSIFICATION_MODEL_PATH = "models/mood_classification_model.h5"
MOOD_TO_SONG_MODEL_PATH = "models/mood_to_song_generator_model.joblib"
DATA_PATH = "data/data_moods.csv"

# Ensure models and data exist
if not os.path.exists(MOOD_CLASSIFICATION_MODEL_PATH):
    st.error(f"Error: Mood classification model not found at {MOOD_CLASSIFICATION_MODEL_PATH}")
    st.stop()
if not os.path.exists(MOOD_TO_SONG_MODEL_PATH):
    st.error(f"Error: Mood-to-song model not found at {MOOD_TO_SONG_MODEL_PATH}")
    st.stop()
if not os.path.exists(DATA_PATH):
    st.error(f"Error: Data file not found at {DATA_PATH}")
    st.stop()

# Load models
mood_classification_model = tf.keras.models.load_model(MOOD_CLASSIFICATION_MODEL_PATH)
mood_to_song_model = joblib.load(MOOD_TO_SONG_MODEL_PATH)

# Load dataset
try:
    df = pd.read_csv(DATA_PATH)
    df['mood'] = df['mood'].str.strip().str.lower()  # Standardize mood column
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# Mapping from facial emotion categories to song mood categories
emotion_to_song_mood = {
    "angry": "Energetic",
    "disgust": "Calm",  
    "fear": "Calm",
    "happy": "Happy",
    "neutral": "Calm",
    "sad": "Sad",
    "surprise": "Energetic"
}

# Define mood classification labels based on training categories
mood_classification_labels = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

def predict_mood(image, model):
    """Preprocess image and predict mood using the classification model."""
    image = image.convert("RGB")  
    image = image.resize((128, 128))  # Resize to match model input shape
    image = np.array(image, dtype=np.float32) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    print("Processed Image Shape:", image.shape)  # Debugging

    try:
        prediction = model.predict(image)
        print("Raw Prediction Output:", prediction)  # Debugging
        predicted_mood_idx = np.argmax(prediction, axis=1)[0]

        detected_emotion = mood_classification_labels[predicted_mood_idx]
        predicted_song_mood = emotion_to_song_mood.get(detected_emotion, "Calm")  # Map emotion to song mood
        
        return detected_emotion, predicted_song_mood  # Return both detected emotion and song mood
    except Exception as e:
        print(f"Prediction Error: {e}")
        return "unknown", "Calm"  # Return default values in case of an error

def recommend_songs(mood, df):
    """Return a sample of recommended songs based on the predicted song mood."""
    mood = mood.lower()
    mood_songs = df[df['mood'] == mood]
    
    if mood_songs.empty:
        return pd.DataFrame({"Message": ["No songs found for this mood."]})  # Return a proper message
    
    return mood_songs[['name', 'artist', 'album']].sample(min(5, len(mood_songs)))


import streamlit as st
from PIL import Image

# Custom CSS for styling
st.markdown(
    """
    <style>
        body {
            background-color: #a1c785 !important; /* Change background color here */
            color: #2e590e; /* Change text color here */
            font-family: Arial, sans-serif;
        }
        .stApp {
        background-color: #a1c785;
        .title {
            color: #2e590e; /* Title color */
            font-size: 28px;
            font-weight: bold;
        }
        .subtitle {
            color: #2e590e; /* Subtitle color */
            font-size: 16px;
        }
        
        .stFileUploader {
            background-color: #a1c785 !important;  /* Change background color */
            border: 2px solid #2e590e !important;  /* Change border color */
            border-radius: 10px !important;  /* Optional: Rounded corners */
            padding: 10px !important;
        }

        
        .stFileUploader > div {
            background-color: #a1c785 !important;  /* Green background */
            color: white !important;  /* White text */
            border-radius: 10px !important;  /* Optional: Rounded corners */
            padding: 10px !important;
        }
        .content-container {
            display: flex;
            color: #a1c785
            align-items: center;
        }
        .uploaded-img img {
            border-radius: 10px;
            width: 150px; /* Adjust image size */
        }
        .result-text {
            font-size: 18px;
            font-weight: bold;
            color: #2e590e; /* Text color for detected mood */
            margin-left: 20px; /* Space between image and text */
        }
        .table-container table {
            background-color: transparent !important; /* Makes table content transparent */
        }

        .table-container th, .table-container td {
            background-color: transparent !important; /* Ensures cells remain transparent */
            color: #2e590e; /* Adjust text color */
            border: 1px solid #064635; /* Keeps borders visible */
        }

    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<p class="title"> 🎵   Mood-Based Song Recommendation</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload an image, and we\'ll recommend songs based on your mood!</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # Displaying image and results in a row
    st.markdown('<div class="content-container">', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(image, caption="Uploaded Image", width=150)

    with col2:
        try:
            detected_emotion, predicted_song_mood = predict_mood(image, mood_classification_model)
            st.markdown(f'<p class="result-text">🎭 Detected Facial Emotion: <b>{detected_emotion.capitalize()}</b></p>', unsafe_allow_html=True)
            st.markdown(f'<p class="result-text">🎶 Mapped Mood for Songs: <b>{predicted_song_mood}</b></p>', unsafe_allow_html=True)
        except:
            pass  # Hides any error message

    st.markdown('</div>', unsafe_allow_html=True)

    # Table of recommended songs
    recommended_songs = recommend_songs(predicted_song_mood, df)
    st.markdown('<div class="table-container">', unsafe_allow_html=True)
    st.markdown('<p class="result-text">🎼 Recommended Songs:</p>', unsafe_allow_html=True)
    st.write(recommended_songs)
    st.markdown('</div>', unsafe_allow_html=True)