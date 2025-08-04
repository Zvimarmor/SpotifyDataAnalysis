from sklearn.metrics import classification_report, confusion_matrix, top_k_accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

from collections import Counter

# ========= SETUP =========
FIG_DIR = "Final_project/figures/model_plots"
os.makedirs(FIG_DIR, exist_ok=True)

TEST_SIZE = 0.25
RANDOM_STATE = 42
HIDDEN_LAYER_SIZES = (256, 128, 64)
EPOCHS = 50
BATCH_SIZE = 64

# ========= LOAD & CLEAN DATA =========
df = pd.read_csv("cleaned_data/total_df.csv")
df = df.drop_duplicates(subset="track_id").dropna()

df = df.rename(columns={
    'popularity_x': 'popularity',
    'duration_ms_x': 'duration_ms',
    'danceability_x': 'danceability',
    'energy_x': 'energy',
    'key_x': 'key',
    'loudness_x': 'loudness',
    'mode_x': 'mode',
    'speechiness_x': 'speechiness',
    'acousticness_x': 'acousticness',
    'instrumentalness_x': 'instrumentalness',
    'liveness_x': 'liveness',
    'valence_x': 'valence',
    'tempo_x': 'tempo',
    'time_signature_x': 'time_signature'
})
columns_to_drop = [
    'track_name_y', 'popularity_y', 'danceability_y', 'energy_y', 'key_y',
    'loudness_y', 'mode_y', 'speechiness_y', 'acousticness_y', 'instrumentalness_y',
    'liveness_y', 'valence_y', 'tempo_y', 'duration_ms_y', 'time_signature_y',
    'Unnamed: 0'
]
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

# ========= FEATURES =========
base_features = ['duration_ms','danceability','energy','key','loudness','mode',
                 'speechiness','acousticness','instrumentalness','liveness',
                 'valence','tempo','time_signature']
df = pd.get_dummies(df, columns=['track_genre'])  # One-hot encode genre
genre_features = [col for col in df.columns if col.startswith('track_genre_')]
features = base_features + genre_features
X = df[features]

# ========= TARGET =========
def popularity_bucket(p):
    return min(int(p // 10), 9)

y_class = df['popularity'].apply(popularity_bucket)

# ========= PREPROCESSING =========
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
y_cat = to_categorical(y_class, num_classes=10)

X_train, X_test, y_train, y_test, y_train_raw, y_test_raw = train_test_split(
    X_scaled, y_cat, y_class, test_size=TEST_SIZE, random_state=RANDOM_STATE)

# ========= CLASS WEIGHTS =========
y_train_int = np.argmax(y_train, axis=1)
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_int), y=y_train_int)
class_weights_dict = dict(enumerate(class_weights))

# ========= KERAS MODEL =========
model = Sequential([
    Dense(HIDDEN_LAYER_SIZES[0], activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.3),
    Dense(HIDDEN_LAYER_SIZES[1], activation='relu'),
    Dropout(0.3),
    Dense(HIDDEN_LAYER_SIZES[2], activation='relu'),
    Dropout(0.3),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
early_stop = EarlyStopping(patience=5, restore_best_weights=True)

history = model.fit(X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    class_weight=class_weights_dict,
                    callbacks=[early_stop],
                    verbose=1)

# ========= EVALUATE KERAS MODEL =========
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nKeras Model Accuracy (Top-1): {acc:.2f}")

y_pred_probs = model.predict(X_test)
y_pred_classes = np.argmax(y_pred_probs, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# --- TOP-3 Accuracy (Keras) ---
top3_accuracy_keras = top_k_accuracy_score(y_true_classes, y_pred_probs, k=3)
print(f"Top-3 Accuracy (Keras): {top3_accuracy_keras:.2f}")

# --- Classification Report ---
print("\n=== Keras Classification Report ===")
print(classification_report(y_true_classes, y_pred_classes))

# --- Confusion Matrix ---
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_true_classes, y_pred_classes), annot=True, fmt='d', cmap='Oranges')
plt.title("Keras Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/keras_confusion_matrix.png")
plt.close()

# ========= BASELINE MODEL =========
most_common_class = Counter(y_train_raw).most_common(1)[0][0]
baseline_preds = np.full_like(y_test_raw, fill_value=most_common_class)

print("\n=== Baseline (Most Frequent Class) Report ===")
print(classification_report(y_test_raw, baseline_preds))

# --- Top-3 for Baseline (dummy probs) ---
baseline_probs = np.zeros((len(baseline_preds), 10))
baseline_probs[np.arange(len(baseline_preds)), baseline_preds] = 1
top3_accuracy_baseline = top_k_accuracy_score(y_test_raw, baseline_probs, k=3)
print(f"Top-3 Accuracy (Baseline): {top3_accuracy_baseline:.2f}")

plt.figure(figsize=(6,5))
sns.heatmap(confusion_matrix(y_test_raw, baseline_preds), annot=True, fmt='d', cmap='Greys')
plt.title("Baseline Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/baseline_confusion_matrix.png")
plt.close()

# ========= RANDOM MODEL =========
random_preds = np.random.randint(0, 10, size=y_test_raw.shape)
print("\n=== Random Model Report ===")
print(classification_report(y_test_raw, random_preds))

# --- Top-3 for Random Model ---
random_probs = np.random.rand(len(random_preds), 10)
random_probs = random_probs / random_probs.sum(axis=1, keepdims=True)
top3_accuracy_random = top_k_accuracy_score(y_test_raw, random_probs, k=3)
print(f"Top-3 Accuracy (Random): {top3_accuracy_random:.2f}")

plt.figure(figsize=(6,5))
sns.heatmap(confusion_matrix(y_test_raw, random_preds), annot=True, fmt='d', cmap='Purples')
plt.title("Random Model Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/random_confusion_matrix.png")
plt.close()
