"""
=====================================================
  Abusive Word Detection - Model Training Script
  Author: B.Tech NLP Mini Project
  Description: Trains a Logistic Regression classifier
               using TF-IDF features for text classification
=====================================================
"""

import pandas as pd
import numpy as np
import pickle
import re
import string
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix
)

# ─────────────────────────────────────────────────
#  STEP 1: Load Dataset
# ─────────────────────────────────────────────────
print("\n[1/5] Loading dataset...")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, "dataset.csv"))

print(f"    Total samples loaded : {len(df)}")
print(f"    Abusive (1)          : {df['label'].sum()}")
print(f"    Not Abusive (0)      : {len(df) - df['label'].sum()}")


# ─────────────────────────────────────────────────
#  STEP 2: Text Preprocessing
# ─────────────────────────────────────────────────
print("\n[2/5] Preprocessing text...")

def preprocess_text(text):
    """
    Clean and normalize input text:
    - Convert to lowercase
    - Remove punctuation
    - Strip extra whitespace
    """
    text = text.lower()                             # lowercase
    text = re.sub(r'\d+', '', text)                 # remove digits
    text = text.translate(
        str.maketrans('', '', string.punctuation)   # remove punctuation
    )
    text = text.strip()                             # strip whitespace
    text = re.sub(r'\s+', ' ', text)                # normalize spaces
    return text

df['cleaned_text'] = df['text'].apply(preprocess_text)
print("    Text preprocessing complete.")


# ─────────────────────────────────────────────────
#  STEP 3: Feature Extraction (TF-IDF)
# ─────────────────────────────────────────────────
print("\n[3/5] Extracting TF-IDF features...")

tfidf = TfidfVectorizer(
    max_features=10000,      # top 10000 words
    ngram_range=(1, 3),      # unigrams + bigrams + trigrams
    sublinear_tf=True,       # apply log normalization
    min_df=1,                # include even rare words
    analyzer='word',
    token_pattern=r'\b[a-zA-Z]{2,}\b'  # only real words
)

X = tfidf.fit_transform(df['cleaned_text'])
y = df['label']

print(f"    Feature matrix shape : {X.shape}")


# ─────────────────────────────────────────────────
#  STEP 4: Train / Test Split & Model Training
# ─────────────────────────────────────────────────
print("\n[4/5] Training Logistic Regression model...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(
    C=5.0,
    solver='lbfgs',
    max_iter=2000,
    class_weight="balanced",
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n    ✔ Model Training Complete!")
print(f"    Accuracy  : {accuracy * 100:.2f}%")
print(f"\n    Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Not Abusive", "Abusive"]))
print(f"    Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ─────────────────────────────────────────────────
#  STEP 5: Save Model & Vectorizer
# ─────────────────────────────────────────────────
print("\n[5/5] Saving model and vectorizer...")

model_dir = os.path.join(BASE_DIR, "model")
os.makedirs(model_dir, exist_ok=True)

with open(os.path.join(model_dir, "model.pkl"), "wb") as f:
    pickle.dump(model, f)

with open(os.path.join(model_dir, "tfidf_vectorizer.pkl"), "wb") as f:
    pickle.dump(tfidf, f)

print(f"    Model saved      → model/model.pkl")
print(f"    Vectorizer saved → model/tfidf_vectorizer.pkl")
print("\n✅ All done! Run `python app.py` to start the Flask server.\n")
