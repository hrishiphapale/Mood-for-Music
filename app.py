import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
from PIL import Image
import pandas as pd
import os

# Define model paths
MOOD_CLASSIFICATION_MODEL_PATH = r"D:\Hrishi DS\Capstone Project\Mood-for-Music\models\mood_classification_model.h5"
MOOD_TO_SONG_MODEL_PATH = r"D:\Hrishi DS\Capstone Project\Mood-for-Music\models\mood_to_song_generator_model.joblib"
DATA_PATH = r"D:\Hrishi DS\Capstone Project\Mood-for-Music\data_moods.csv"

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

# Streamlit UI
st.title("Mood-Based Song Recommendation")
st.write("Upload an image, and we'll recommend songs based on your mood!")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    try:
        detected_emotion, predicted_song_mood = predict_mood(image, mood_classification_model)
        st.write(f"**Detected Facial Emotion:** {detected_emotion.capitalize()}")
        st.write(f"**Mapped Mood for Songs:** {predicted_song_mood}")

        recommended_songs = recommend_songs(predicted_song_mood, df)
        st.write("**Recommended Songs:**")
        st.write(recommended_songs)
    
    except Exception as e:
        st.error(f"An error occurred: {e}")
    