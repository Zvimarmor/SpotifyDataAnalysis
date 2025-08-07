import numpy as np
import librosa
import json
import pickle
import os
from tensorflow.keras.models import load_model
from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials

# ======== SETTINGS ========
MODEL_PATH = "Final_Project/user_interface/model_weights.h5"
SCALER_PATH = "Final_Project/user_interface/scaler.pkl"
GENRE_LIST_PATH = "Final_Project/user_interface/genre_list.json"

# Spotify API credentials (replace with your own) 
# currently set to empty strings (due to the fact we are students and it costs money to use the API)
SPOTIFY_CLIENT_ID = ""
SPOTIFY_CLIENT_SECRET = ""

# ======== LOAD ASSETS ========
model = load_model(MODEL_PATH)
scaler = pickle.load(open(SCALER_PATH, "rb"))
with open(GENRE_LIST_PATH) as f:
    genre_list = json.load(f)


# ======== FEATURE EXTRACTION ========
def get_spotify_audio_features(track_id):
    try:
        sp = Spotify(auth_manager=SpotifyClientCredentials(
            client_id=SPOTIFY_CLIENT_ID,
            client_secret=SPOTIFY_CLIENT_SECRET
        ))
        features = sp.audio_features([track_id])[0]
        if features is None:
            raise ValueError("Track not found or invalid")

        return {
            "duration_ms": features["duration_ms"],
            "tempo": features["tempo"],
            "loudness": features["loudness"],
            "energy": features["energy"],
            "danceability": features["danceability"],
            "speechiness": features["speechiness"],
            "acousticness": features["acousticness"],
            "instrumentalness": features["instrumentalness"],
            "liveness": features["liveness"],
            "valence": features["valence"],
            "key": features["key"],
            "mode": features["mode"],
            "time_signature": features["time_signature"]
        }

    except Exception as e:
        print(f"[WARN] Spotify feature extraction failed: {e}")
        return None

def extract_audio_features_librosa(mp3_path):
    y, sr = librosa.load(mp3_path, duration=30)

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    rms = librosa.feature.rms(y=y).mean()
    spectral_flatness = librosa.feature.spectral_flatness(y=y).mean()
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
    zero_crossings = librosa.feature.zero_crossing_rate(y).mean()

    return {
        "duration_ms": 30000,
        "tempo": tempo,
        "loudness": -20 * np.log10(rms + 1e-6),
        "energy": rms,
        "danceability": spectral_flatness,
        "speechiness": spectral_flatness,
        "acousticness": centroid,
        "instrumentalness": zero_crossings,
        "liveness": bandwidth,
        "valence": rolloff,
        "key": 0,
        "mode": 1,
        "time_signature": 4
    }

# ======== PREPARE INPUT ========
def prepare_input(features_dict, genre='pop'):
    base_features = ['duration_ms', 'danceability', 'energy', 'key', 'loudness',
                     'mode', 'speechiness', 'acousticness', 'instrumentalness',
                     'liveness', 'valence', 'tempo', 'time_signature']
    
    x_base = [features_dict[feat] for feat in base_features]
    genre_vector = [1 if g == genre else 0 for g in genre_list]

    x_full = np.array(x_base + genre_vector).reshape(1, -1)
    x_scaled = scaler.transform(x_full)
    return x_scaled

# ======== MAIN PREDICTOR ========
def predict_popularity_range(mp3_path, genre="pop", spotify_track_id=None):
    features = None

    if spotify_track_id:
        features = get_spotify_audio_features(spotify_track_id)
    
    if features is None:
        print("[INFO] Falling back to librosa feature extraction.")
        print("[WARN] The model may not perform optimally without Spotify features.")
        features = extract_audio_features_librosa(mp3_path)

    x_input = prepare_input(features, genre)
    pred_probs = model.predict(x_input)
    pred_class = np.argmax(pred_probs)

    return f"Predicted Popularity Range: {pred_class*10}–{(pred_class+1)*10 - 1}"

# ======== USAGE EXAMPLE ========
if __name__ == "__main__":
    mp3 = "example_song.mp3"
    track_id = "4qPNDBW1i3p13qLCt0Ki3A"  # Optional: Spotify track ID
    print(predict_popularity_range(mp3, genre="acoustic", spotify_track_id=track_id))
