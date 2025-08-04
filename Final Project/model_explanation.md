# Spotify Track Popularity Prediction

This project explores the prediction of **track popularity levels** on Spotify using machine learning methods. Rather than regressing a continuous score (0–100), the task is formulated as a **multi-class classification problem** with 10 discrete popularity brackets.

---

## Problem Setup

We classify each song into one of 10 popularity classes:

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

This discretization allows the model to focus on general trends rather than precise numerical prediction, while still enabling approximate popularity estimation.

---

## Dataset Assumptions

- Each row represents a single song.
- Songs are described by audio, structural, and categorical features.
- The `popularity` column is a score from 0 to 100 assigned by Spotify.
- Tracks with missing values or multiple artists were excluded.
- The `track_genre` field is a key categorical feature in prediction.

---

## Preprocessing Pipeline

- Duplicate rows and missing data were removed.
- All numerical features were standardized using `StandardScaler`.
- Categorical genre data was encoded using one-hot encoding.
- Popularity was bucketed into classes 0 through 9.

---

## Neural Network Architecture (Keras)

A feedforward neural network was constructed with the following design:

- **Input:** Standardized numerical and one-hot categorical features.
- **Hidden Layers:** Three fully connected layers with sizes `[256, 128, 64]` and ReLU activations.
- **Regularization:** Dropout layers (rate = 0.3) follow each hidden layer.
- **Output:** A softmax layer with 10 units (for each popularity class).
- **Loss Function:** Categorical cross-entropy.
- **Optimizer:** Adam.
- **Training Strategy:**
  - Class weights are computed to account for class imbalance.
  - Early stopping is used with a patience of 5 epochs to prevent overfitting.

---

## Additional Models for Benchmarking

To validate model performance, two classical classifiers were implemented using the same feature set:

- **Decision Tree Classifier**
- **K-Nearest Neighbors (KNN)**

These serve as interpretable and efficient baselines for comparison.

---

## Baseline Models

Two non-learned baselines were constructed:

### 1. Most Frequent Class Baseline

This model always predicts the most common class in the training data (class 2 in our case). Its performance reflects a naive strategy that ignores input features.

**Performance Summary:**

- Accuracy: 0.20  
- Weighted F1-score: 0.06  
- Precision and recall for all other classes: near zero

### 2. Random Class Baseline

This model randomly samples a class (0–9) for each prediction with equal probability. It serves as a reference for evaluating whether a model captures any structure at all.

**Performance Summary:**

- Accuracy: 0.10  
- Weighted F1-score: 0.12  
- Performance uniformly low across all classes

---

## Evaluation Metrics

- **Top-1 Accuracy:** Whether the highest probability prediction matches the true class.
- **Top-3 Accuracy:** Whether the true class is among the three most probable predictions (by softmax ranking).

Top-3 accuracy provides a relaxed measure of model confidence and proximity to correct prediction.

---

## Final Results

| Model                      | Top-1 Accuracy | Top-3 Accuracy |
|---------------------------|----------------|----------------|
| Neural Network (Keras)    | 0.46           | 0.80           |
| K-Nearest Neighbors        | 0.51           | N/A            |
| Decision Tree              | 0.45           | N/A            |
| Baseline (most frequent)   | 0.20           | N/A            |
| Baseline (random guessing) | 0.10           | N/A            |

Despite the slightly higher Top-1 accuracy of KNN, the neural network showed significantly better Top-3 performance and recall across rare classes, suggesting improved generalization.

---

## Project Structure

- `model.py`: Main training and evaluation script
- `cleaned_data/Spotify_Tracks_lone_artists.csv`: Preprocessed input data
- `model_explanation.md`: This documentation file
- `requirements.txt`: Python dependencies

---

# חיזוי רמת הפופולריות של שירים בספוטיפיי

פרויקט זה עוסק בחיזוי **רמות פופולריות של שירים** בספוטיפיי באמצעות למידת מכונה. במקום לנבא ציון רציף בין 0 ל־100, המשימה מוגדרת כבעיית סיווג לעשר קטגוריות פופולריות נפרדות.

---

## הגדרת הבעיה

מטרת המודל היא לשייך כל שיר לקטגוריה מתוך 10 טווחים:

| קטגוריה | טווח פופולריות |
|----------|----------------|
| 0        | 0–9            |
| 1        | 10–19          |
| 2        | 20–29          |
| 3        | 30–39          |
| 4        | 40–49          |
| 5        | 50–59          |
| 6        | 60–69          |
| 7        | 70–79          |
| 8        | 80–89          |
| 9        | 90–100         |

סיווג זה מפשט את בעיית החיזוי ומאפשר למידה על מגמות כלליות של פופולריות.

---

## הנחות על הדאטה

- כל שורה מייצגת שיר בודד.
- הנתונים כוללים מאפיינים אקוסטיים וקטגוריים של השיר.
- עמודת `popularity` היא ציון בין 0 ל־100 שניתן על ידי ספוטיפיי.
- הוסרו שירים עם ערכים חסרים או ריבוי אמנים.
- השדה `track_genre` הוא תכונה קטגורית חשובה במיוחד.

---

## תהליך עיבוד מקדים

- הוסרו כפילויות ושורות עם ערכים חסרים.
- תכונות נומריות עברו סטנדרטיזציה עם `StandardScaler`.
- קידוד `track_genre` בוצע באמצעות one-hot encoding.
- ציון הפופולריות הומר לקטגוריה בין 0 ל־9 לפי עשירונים.

---

## ארכיטקטורת רשת עצבית (Keras)

מודל רשת עצבית נבנה עם המאפיינים הבאים:

- **קלט:** פיצ'רים נומריים וקטגוריים תקניים.
- **שכבות חבויות:** שלוש שכבות צפופות `[256, 128, 64]` עם הפעלת ReLU.
- **רגולריזציה:** Dropout של 0.3 לאחר כל שכבה.
- **פלט:** שכבת softmax עם 10 קטגוריות.
- **פונקציית הפסד:** categorical_crossentropy.
- **אופטימיזציה:** Adam.
- **אמצעי שיפור:** משקלי מחלקות והתניה מוקדמת (`EarlyStopping`).

---

## מודלים להשוואה

הושוו גם שני מודלים קלאסיים נוספים:

- **Decision Tree**
- **K-Nearest Neighbors**

כל המודלים עברו על אותם פיצ'רים ובאותה חלוקת אימון/בדיקה.

---

## מודלים בסיסיים (Baseline)

### 1. מודל תווית שכיחה

מודל זה מנבא תמיד את הקטגוריה השכיחה ביותר ב־Train (class 2). מטרתו להוות קו השוואה לביצועים נאיביים.

**ביצועים:**

- דיוק: 0.20  
- F1 ממוצע משוקלל: 0.06  
- שאר המדדים: כמעט אפס

### 2. מודל אקראי

מודל זה בוחר קטגוריה באקראי לחלוטין לכל שיר. הוא מייצג אסטרטגיה ללא כל למידה מהדאטה.

**ביצועים:**

- דיוק: 0.10  
- F1 ממוצע משוקלל: 0.12  
- מדדים נמוכים ואחידים בכל הקטגוריות

---

## מדדי הערכה

- **Top-1 Accuracy**: מדד דיוק רגיל – האם החיזוי תואם לתווית.
- **Top-3 Accuracy**: האם התווית הנכונה הופיעה בין שלוש ההסתברויות הגבוהות ביותר של המודל (לפי softmax).

Top-3 מאפשר לבחון את קרבת המודל לתוצאה נכונה, גם אם לא מדויקת.

---

## תוצאות סופיות

| מודל                       | Top-1 Accuracy | Top-3 Accuracy |
|----------------------------|----------------|----------------|
| רשת נוירונים (Keras)       | 0.46           | 0.80           |
| K-Nearest Neighbors         | 0.51           | N/A            |
| Decision Tree               | 0.45           | N/A            |
| בסיס (קטגוריה שכיחה)       | 0.20           | N/A            |
| בסיס רנדומלי               | 0.10           | N/A            |

למרות יתרון קל ב־Top-1 עבור KNN, המודל הנוירוני השיג תוצאות טובות יותר בממוצע, במיוחד בקטגוריות נדירות וב־Top-3, דבר שמרמז על למידה כללית עמוקה יותר.

---

## מבנה התיקיה

- `model.py`: קובץ האימון והערכת המודל הראשי
- `cleaned_data/Spotify_Tracks_lone_artists.csv`: קובץ הנתונים
- `model_explanation.md`: קובץ תיעוד זה
- `requirements.txt`: קובץ ספריות
