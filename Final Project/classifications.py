import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# ========= FIGURE OUTPUT =========
FIG_DIR = "Final Project/figures"
os.makedirs(FIG_DIR, exist_ok=True)

# ========= LOAD DATA =========
df = pd.read_csv("cleaned_data/Spotify_Tracks_lone_artists.csv")
df = df.drop_duplicates(subset="track_id").dropna()
df = pd.get_dummies(df, columns=['track_genre'])

# ========= FEATURE ENGINEERING =========
base_features = ['duration_ms','danceability','energy','key','loudness','mode',
                 'speechiness','acousticness','instrumentalness','liveness',
                 'valence','tempo','time_signature']
genre_features = [col for col in df.columns if col.startswith('track_genre_')]
features = base_features + genre_features
X = df[features]

def popularity_bucket(p):
    return min(int(p // 10), 9)

y_raw_class = df['popularity'].apply(popularity_bucket)

# ========= SCALING =========
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ========= SPLIT =========
TEST_SIZE = 0.25
RANDOM_STATE = 42

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_scaled, y_raw_class, test_size=TEST_SIZE, random_state=RANDOM_STATE)

# ========= CLASSICAL MODELS =========

# --- Decision Tree ---
tree = DecisionTreeClassifier(random_state=RANDOM_STATE)
tree.fit(X_train_c, y_train_c)
tree_preds = tree.predict(X_test_c)

# --- KNN ---
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_c, y_train_c)
knn_preds = knn.predict(X_test_c)

# --- Reports ---
print("\n=== Decision Tree Classification Report ===")
print(classification_report(y_test_c, tree_preds))

print("\n=== KNN Classification Report ===")
print(classification_report(y_test_c, knn_preds))

# --- Confusion Matrices ---
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(confusion_matrix(y_test_c, tree_preds), annot=True, fmt='d', cmap='Blues', ax=axs[0])
axs[0].set_title("Decision Tree Confusion Matrix")
axs[0].set_xlabel("Predicted")
axs[0].set_ylabel("True")

sns.heatmap(confusion_matrix(y_test_c, knn_preds), annot=True, fmt='d', cmap='Greens', ax=axs[1])
axs[1].set_title("KNN Confusion Matrix")
axs[1].set_xlabel("Predicted")
axs[1].set_ylabel("True")

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/classical_models_confusion_matrices.png")
plt.close()
