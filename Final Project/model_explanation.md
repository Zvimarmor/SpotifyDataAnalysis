# Spotify Track Popularity Prediction

This project focuses on predicting the **popularity level** of a song on Spotify using machine learning. Instead of regressing exact popularity scores (0–100), we divide popularity into 10 discrete brackets and approach the task as a **multi-class classification problem**.

---

## Problem Setup

We aim to classify each song into one of 10 popularity buckets:

| Class | Popularity Range |
|-------|------------------|
| 0     | 0–9              |
| 1     | 10–19            |
| 2     | 20–29            |
| 3     | 30–39            |
| 4     | 40–49            |
| 5     | 50–59            |
| 6     | 60–69            |
| 7     | 70–79            |
| 8     | 80–89            |
| 9     | 90–100           |

This approach simplifies the complexity of predicting an exact score, while still allowing us to evaluate how "close" predictions are.

---

## Dataset Assumptions

We assume the following about the dataset:

- Each row represents a single track.
- Each track has acoustic and structural features (e.g., tempo, energy, danceability, etc.).
- The target variable is Spotify's `popularity` score (integer from 0 to 100).
- Tracks with multiple artists or missing fields are removed during preprocessing.
- The `track_genre` field is one of the most influential categorical variables.

---

## Preprocessing

- Dropped duplicates and missing values.
- Standardized all numeric features using `StandardScaler`.
- Applied one-hot encoding to the `track_genre` categorical feature.
- Converted the continuous popularity score into a class label (0–9) using integer bucketing.

---

## Neural Network Architecture (Keras)

- Input: All standardized numeric and one-hot categorical features.
- Hidden layers: 3 dense layers with sizes `[256, 128, 64]` and ReLU activations.
- Regularization: Dropout layers (rate = 0.3) after each dense layer.
- Output: 10-class softmax layer.
- Loss: Categorical cross-entropy.
- Optimizer: Adam.
- Training enhancements:
  - Class weights calculated based on class distribution (to mitigate imbalance).
  - Early stopping based on validation loss (patience = 5).

---

## Additional Classifiers for Comparison

To benchmark the neural network, we implemented:

- **Decision Tree Classifier**
- **K-Nearest Neighbors (KNN)**

All models were evaluated on the same train/test split with the same feature set.

---

## Baseline Model

We implemented a simple naive baseline model which always predicts the **most common class** in the training set (e.g., class 2). This helps determine whether our classifiers are learning meaningful patterns beyond simple class frequency.

Performance of the baseline model:

- Accuracy: 0.20
- F1-score (weighted): 0.06
- Precision and recall for all other classes: 0.00

---

## Evaluation Metrics

We used the following metrics:

- **Top-1 Accuracy**: Standard accuracy – whether the top predicted class matches the true class.
- **Top-3 Accuracy**: Whether the correct class appears in the top 3 predicted probabilities (by softmax).

This allows us to evaluate not only exact correctness but also the model's ability to narrow down to the right region of popularity.

---

## Final Results

| Model               | Top-1 Accuracy | Top-3 Accuracy |
|--------------------|----------------|----------------|
| Neural Network (Keras) | 0.46           | 0.80           |
| Decision Tree       | 0.45           | N/A            |
| K-Nearest Neighbors | 0.51           | N/A            |
| Baseline (most common class) | 0.20    | N/A            |

Note: While KNN slightly outperformed the neural network in Top-1, the neural model had far better precision in rare classes and achieved **very strong Top-3 accuracy**, demonstrating good generalization of "near-miss" predictions.

---

## File Structure

- `model.py`: Main training and evaluation script
- `cleaned_data/Spotify_Tracks_lone_artists.csv`: Cleaned input dataset
- `model_explanation.md`: Documentation
- `requirements.txt`: Python dependencies





# חיזוי רמת הפופולריות של שירים בספוטיפיי

מטרת הפרויקט היא לחזות את **רמת הפופולריות** של שיר בספוטיפיי באמצעות מודלים של למידת מכונה. במקום לחזות את ציון הפופולריות המדויק (0–100), חילקנו את התחום ל־10 קטגוריות (עשירונים) והתייחסנו למשימה כאל **בעיית סיווג רב־מחלקתי (multi-class classification)**.

---

## הגדרת הבעיה

כל שיר מקבל תווית פופולריות על פי הטווח הבא:

| קטגוריה | טווח פופולריות |
|----------|-----------------|
| 0        | 0–9             |
| 1        | 10–19           |
| 2        | 20–29           |
| 3        | 30–39           |
| 4        | 40–49           |
| 5        | 50–59           |
| 6        | 60–69           |
| 7        | 70–79           |
| 8        | 80–89           |
| 9        | 90–100          |

גישה זו מפשטת את החיזוי ומאפשרת למדוד הצלחה לפי קרבה לרמת הפופולריות.

---

## הנחות על הדאטה

- כל שורה מייצגת שיר יחיד.
- הנתונים כוללים תכונות אקוסטיות ומבניות של השיר (כמו קצב, אנרגיה, ועוד).
- ציון הפופולריות הוא של ספוטיפיי (בין 0 ל־100).
- שירים עם שדות חסרים או ריבוי אמנים הוסרו.
- השדה `track_genre` נחשב לתכונה קטגורית משמעותית במיוחד.

---

## עיבוד מקדים

- סינון כפילויות ושורות עם ערכים חסרים.
- נרמול של תכונות נומריות באמצעות `StandardScaler`.
- המרה של `track_genre` ל־one-hot encoding.
- המרה של הפופולריות לקטגוריות לפי עשירונים.

---

## ארכיטקטורת רשת עצבית (Keras)

- קלט: כלל הפיצ'רים הנומריים והקטגוריים.
- שכבות חבויות: `[256, 128, 64]` עם `ReLU`.
- רגולריזציה: `Dropout` של 0.3 לאחר כל שכבה.
- שכבת פלט: `softmax` עם 10 תוויות.
- פונקציית הפסד: `categorical_crossentropy`.
- אופטימיזר: Adam.
- שיפורים:
  - משקלי מחלקות אוטומטיים (לטיפול באי־איזון).
  - עצירת אימון מוקדמת (`EarlyStopping`) לפי `val_loss`.

---

## מודלים להשוואה

לצורך השוואה, מימשנו גם:

- Decision Tree Classifier
- K-Nearest Neighbors (KNN)

שני המודלים משתמשים באותם פיצ'רים ובאותו חלוקה ל־Train/Test.

---

## מודל בסיס (Baseline)

מימשנו מודל נאיבי שמנבא תמיד את הקטגוריה הנפוצה ביותר (לרוב class 2). כך נוכל להשוות מול מודלים "חכמים".

תוצאות מודל הבסיס:

- דיוק (Accuracy): 0.20
- F1 ממוצע משוקלל: 0.06
- precision ו־recall בשאר הקבוצות: 0.00

---

## מדדי הערכה

- **Top-1 Accuracy** – האם התווית שנחזתה היא בדיוק הנכונה.
- **Top-3 Accuracy** – האם התווית הנכונה הופיעה בין שלוש התחזיות עם ההסתברות הגבוהה ביותר (לפי softmax).

Top-3 מאפשר הערכה רכה של "קרבה נכונה", גם כשאין דיוק מוחלט.

---

## תוצאות סופיות

| מודל                | Top-1 Accuracy | Top-3 Accuracy |
|---------------------|----------------|----------------|
| רשת נוירונים (Keras) | 0.46           | 0.80           |
| Decision Tree        | 0.45           | N/A            |
| K-Nearest Neighbors  | 0.51           | N/A            |
| מודל בסיס (class 2)  | 0.20           | N/A            |

הרשת הנוירונית השיגה תוצאות טובות מאוד, במיוחד במדדי Top-3, וגם ידעה לחזות היטב קטגוריות נדירות – דבר שמודל הבסיס כשל בו.

---

## מבנה התיקיה

- `model.py` – קובץ ההרצה הראשי
- `cleaned_data/Spotify_Tracks_lone_artists.csv` – קובץ הדאטה הנקי
- `model_explanation.md` – קובץ תיעוד זה
- `requirements.txt` – קובץ ספריות דרושות

