import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

from collections import Counter

# ========= SETUP FIGURE OUTPUT DIRECTORY =========
FIG_DIR = "Final Project/figures"
os.makedirs(FIG_DIR, exist_ok=True)

# ========= HYPERPARAMETERS =========
TEST_SIZE = 0.25
RANDOM_STATE = 42
HIDDEN_LAYER_SIZES = (256, 128, 64)
EPOCHS = 50
BATCH_SIZE = 64

# ========= LOAD DATA =========
df = pd.read_csv("cleaned_data/Spotify_Tracks_lone_artists.csv")
df = df.drop_duplicates(subset="track_id").dropna()

# ========= FEATURE ENGINEERING =========
df = pd.get_dummies(df, columns=['track_genre'])

base_features = ['duration_ms','danceability','energy','key','loudness','mode',
                 'speechiness','acousticness','instrumentalness','liveness',
                 'valence','tempo','time_signature']
genre_features = [col for col in df.columns if col.startswith('track_genre_')]
features = base_features + genre_features
X = df[features]

# Create target: popularity bucket (0–9)
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
model = Sequential()
model.add(Dense(HIDDEN_LAYER_SIZES[0], activation='relu', input_shape=(X.shape[1],)))
model.add(Dropout(0.3))
model.add(Dense(HIDDEN_LAYER_SIZES[1], activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(HIDDEN_LAYER_SIZES[2], activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

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

# ========= TOP-3 ACCURACY =========
top3_preds = np.argsort(y_pred_probs, axis=1)[:, -3:]
correct_top3 = np.any(top3_preds == y_true_classes.reshape(-1, 1), axis=1)
top3_accuracy = np.mean(correct_top3)
print(f"Top-3 Accuracy: {top3_accuracy:.2f}")

# ========= REPORT =========
print("\n=== Keras Classification Report ===")
print(classification_report(y_true_classes, y_pred_classes))

# ========= CONFUSION MATRIX =========
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_true_classes, y_pred_classes), annot=True, fmt='d', cmap='Oranges')
plt.title("Keras Confusion Matrix (10-Class Popularity)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/keras_confusion_matrix.png")
plt.close()

# ========= TRAINING CURVE =========
plt.figure()
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training Accuracy over Epochs")
plt.legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/keras_training_accuracy.png")
plt.close()

# ========= BASELINE MODEL =========
most_common_class = Counter(y_train_raw).most_common(1)[0][0]
baseline_preds = np.full_like(y_test_raw, fill_value=most_common_class)

print("\n=== Baseline (Most Frequent Class) Report ===")
print(f"Most Frequent Class: {most_common_class}")
print(classification_report(y_test_raw, baseline_preds))

plt.figure(figsize=(6,5))
sns.heatmap(confusion_matrix(y_test_raw, baseline_preds), annot=True, fmt='d', cmap='Greys')
plt.title("Baseline Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/baseline_confusion_matrix.png")
plt.close()

# ========= CLASSICAL MODELS FOR COMPARISON =========

# Prepare data for classical models
y_raw_class = df['popularity'].apply(popularity_bucket)

# Train-test split
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_scaled, y_raw_class, test_size=TEST_SIZE, random_state=RANDOM_STATE)

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
