# Spotify Popularity Predictor – User Interface

This module provides a graphical user interface (GUI) that allows users to predict the popularity range of a song using a pre-trained neural network model. The prediction is based on audio features extracted from an MP3 file and a selected musical genre.

## How to Launch the Interface

To start the app, make sure you're in the project root directory and that your virtual environment is activated.

Run:

```bash
python Final_Project/user_interface/predict_gui.py


---

## How to Use

1. Click the **Browse** button to select an MP3 file from your computer.
2. Choose a genre from the dropdown list.
3. Click the **Predict Popularity** button.
4. A popup will display the predicted popularity range (e.g., "Predicted Popularity Range: 30–39").

---

## How It Works

- The app uses `librosa` to extract audio features from the MP3 (such as tempo, loudness, danceability, etc.).
- The genre is one-hot encoded based on the list in `genre_list.json`.
- All features are standardized using a saved `StandardScaler` (`scaler.pkl`).
- A pre-trained Keras model (`model_weights.h5`) outputs a probability distribution across 10 popularity buckets (0–9).
- The class with the highest probability determines the predicted range (e.g., class 5 → 50–59).

---

## File Structure

- `predict_gui.py`: The GUI application (tkinter).
- `predict_popularity.py`: Contains logic for feature extraction, scaling, and model inference.
- `model_weights.h5`: The trained Keras model.
- `scaler.pkl`: The feature scaler used in training.
- `genre_list.json`: List of available genres for one-hot encoding.

---

## Notes

- MP3 files must be valid and contain at least 30 seconds of audio.
- This app was trained on Spotify data; generalization to other sources may vary.

---

## Authors

Created by Zvi Marmor, Shaked Hartal, and Shai Abu  
as part of the "A Needle in a Data Haystack" course at the Hebrew University of Jerusalem.
