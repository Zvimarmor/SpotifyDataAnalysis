import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ========= HYPERPARAMETERS =========
TEST_SIZE = 0.2
RANDOM_STATE = 42
HIDDEN_LAYER_SIZES = (64, 64)
LEARNING_RATE = 0.001
MAX_ITER = 300

# ========= LOAD DATA =========
df = pd.read_csv("cleaned_data/Spotify_Tracks_lone_artists.csv")

# ========= CLEAN & PREP =========
df = df.drop_duplicates(subset="track_id").dropna()

# Features to use
features = ['duration_ms','danceability','energy','key','loudness','mode',
            'speechiness','acousticness','instrumentalness','liveness',
            'valence','tempo','time_signature']
X = df[features]
y = df['popularity']

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

# ========= REGRESSION MODEL =========
regressor = MLPRegressor(hidden_layer_sizes=HIDDEN_LAYER_SIZES,
                         learning_rate_init=LEARNING_RATE,
                         max_iter=MAX_ITER,
                         random_state=RANDOM_STATE)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

# ========= EVALUATE REGRESSION =========
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n=== Regressor Performance ===")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")

# ========= PLOT RESULTS =========
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(regressor.loss_curve_)
plt.title("Loss Curve")
plt.xlabel("Iterations")
plt.ylabel("Loss")

plt.subplot(1,2,2)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("True Popularity")
plt.ylabel("Predicted Popularity")
plt.title(f"Predicted vs True (R² = {r2:.2f})")

plt.tight_layout()
plt.show()

# ========= CLASSIFICATION =========
# Convert popularity to classes: low (0), medium (1), high (2)
def pop_to_class(p):
    if p < 50:
        return 0
    elif p < 75:
        return 1
    else:
        return 2

y_class = df['popularity'].apply(pop_to_class)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_scaled, y_class, test_size=TEST_SIZE, random_state=RANDOM_STATE)

# Decision Tree Classifier
tree = DecisionTreeClassifier(random_state=RANDOM_STATE)
tree.fit(X_train_c, y_train_c)
tree_preds = tree.predict(X_test_c)

# KNN Classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_c, y_train_c)
knn_preds = knn.predict(X_test_c)

# Reports
print("\n=== Decision Tree Classification Report ===")
print(classification_report(y_test_c, tree_preds))

print("\n=== KNN Classification Report ===")
print(classification_report(y_test_c, knn_preds))

# Confusion matrices
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
plt.show()
