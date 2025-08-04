import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, top_k_accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# ========= FIGURE OUTPUT =========
FIG_DIR = "Final_project/figures/model_plots"
os.makedirs(FIG_DIR, exist_ok=True)

# ========= LOAD DATA =========
df = pd.read_csv("cleaned_data/total_df.csv")
df = df.drop_duplicates(subset="track_id").dropna()

# Keep only necessary columns and rename feature columns from *_x
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
df = pd.get_dummies(df, columns=['track_genre'])

# ========= FEATURES =========
base_features = ['duration_ms','danceability','energy','key','loudness','mode',
                 'speechiness','acousticness','instrumentalness','liveness',
                 'valence','tempo','time_signature']
genre_features = [col for col in df.columns if col.startswith('track_genre_')]
features = base_features + genre_features
X = df[features]

# ========= TARGET =========
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
tree_probs = tree.predict_proba(X_test_c)
top3_tree = top_k_accuracy_score(y_test_c, tree_probs, k=3)

# --- KNN ---
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_c, y_train_c)
knn_preds = knn.predict(X_test_c)
knn_probs = knn.predict_proba(X_test_c)
top3_knn = top_k_accuracy_score(y_test_c, knn_probs, k=3)

# --- Reports ---
print("\n=== Decision Tree Classification Report ===")
print(classification_report(y_test_c, tree_preds))
print(f"Top-3 Accuracy (Decision Tree): {top3_tree:.2f}")

print("\n=== KNN Classification Report ===")
print(classification_report(y_test_c, knn_preds))
print(f"Top-3 Accuracy (KNN): {top3_knn:.2f}")

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
