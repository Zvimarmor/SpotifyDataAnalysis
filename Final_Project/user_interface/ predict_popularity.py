import numpy as np
import librosa
import json
import pickle
from tensorflow.keras.models import load_model

# ========= LOAD ASSETS =========
MODEL_PATH = "Final_project/user_interface/model_weights.h5"
SCALER_PATH = "Final_project/user_interface/scaler.pkl"
GENRE_LIST_PATH = "Final_project/user_interface/genre_list.json"

model = load_model(MODEL_PATH)
scaler = pickle.load(open(SCALER_PATH, "rb"))
with open(GENRE_LIST_PATH) as f:
    genre_list = json.load(f)

# ========= HELPER: Extract Features =========
def extract_audio_features(mp3_path):
    y, sr = librosa.load(mp3_path, duration=30)
    features = {
        "duration_ms": 30_000,  # assume 30 sec preview
        "tempo": librosa.beat.tempo(y=y, sr=sr)[0],
        "loudness": librosa.feature.rms(y=y).mean(),
        "energy": librosa.feature.rms(y=y).mean(),
        "danceability": librosa.feature.tempogram(y=y, sr=sr).mean(),
        "speechiness": librosa.feature.spectral_flatness(y=y).mean(),
        "acousticness": librosa.feature.spectral_centroid(y=y, sr=sr).mean(),
        "instrumentalness": librosa.feature.zero_crossing_rate(y=y).mean(),
        "liveness": librosa.feature.spectral_bandwidth(y=y, sr=sr).mean(),
        "valence": librosa.feature.spectral_rolloff(y=y, sr=sr).mean(),
        "key": 0,  # placeholder
        "mode": 1, # placeholder
        "time_signature": 4  # assumption
    }
    return features

# ========= HELPER: Format into Model Input =========
def prepare_input(features_dict, genre='pop'):
    base_features = ['duration_ms', 'danceability', 'energy', 'key', 'loudness',
                     'mode', 'speechiness', 'acousticness', 'instrumentalness',
                     'liveness', 'valence', 'tempo', 'time_signature']
    x_base = [features_dict[feat] for feat in base_features]

    # Genre one-hot
    genre_vector = [1 if g == genre else 0 for g in genre_list]
    
    x_full = np.array(x_base + genre_vector).reshape(1, -1)
    x_scaled = scaler.transform(x_full)
    return x_scaled

# ========= MAIN PREDICTION =========
def predict_popularity_range(mp3_path, genre="pop"):
    features = extract_audio_features(mp3_path)
    x_input = prepare_input(features, genre)
    pred_probs = model.predict(x_input)
    pred_class = np.argmax(pred_probs)
    return f"Predicted Popularity Range: {pred_class*10}–{(pred_class+1)*10 - 1}"

# ========= EXAMPLE USAGE =========
if __name__ == "__main__":
    mp3 = "example_song.mp3"
    print(predict_popularity_range(mp3, genre="pop"))
