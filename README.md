# 🛡️ Abusive Word Detection using NLP

## 📌 Project Overview

**GuardNLP** is a web-based text classification system that detects whether a given word, phrase, or sentence is **Abusive** or **Not Abusive**. It uses a trained **Logistic Regression** model on **TF-IDF** features, exposed via a **Flask REST API**, and served through a clean, modern frontend.

Live Demo :[https://abusive-word-detector-17g5.onrender.com](https://abusive-word-detector-17g5.onrender.com)

## 🗂️ Project Structure

```
abusive-word-detector/
│
├── app.py                    # Flask backend — API server
├── train_model.py            # ML training script
├── dataset.csv               # Sample labeled dataset
├── requirements.txt          # Python dependencies
│
├── model/
│   ├── model.pkl             # Saved trained model (generated)
│   └── tfidf_vectorizer.pkl  # Saved TF-IDF vectorizer (generated)
│
├── templates/
│   └── index.html            # Frontend HTML page
│
└── static/
    ├── css/
    │   └── style.css         # Stylesheet
    └── js/
        └── main.js           # Frontend JavaScript logic
```

---

## ⚙️ Tech Stack

| Layer     | Technology                      |
|-----------|----------------------------------|
| Frontend  | HTML5, CSS3, Vanilla JavaScript  |
| Backend   | Python, Flask, Flask-CORS        |
| NLP/ML    | scikit-learn (TF-IDF + LogReg)   |
| Data      | pandas, numpy                    |

---

## 🧠 ML Pipeline

```
Raw Text
   ↓
Preprocessing (lowercase, remove punctuation/digits)
   ↓
TF-IDF Vectorization (unigrams + bigrams, max 5000 features)
   ↓
Logistic Regression Classifier
   ↓
Prediction + Confidence Score
```

---

## 🚀 How to Run

### Step 1 — Clone & Setup

```bash
git clone <your-repo-url>
cd abusive-word-detector
```

### Step 2 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Train the Model

```bash
python train_model.py
```

This will create `model/model.pkl` and `model/tfidf_vectorizer.pkl`.

### Step 4 — Start the Flask Server

```bash
python app.py
```

### Step 5 — Open in Browser

```
http://127.0.0.1:5000
```

---

## 🔌 API Reference

### `POST /predict`

Predicts whether input text is abusive.

**Request:**
```json
{
  "text": "you are so stupid"
}
```

**Response:**
```json
{
  "label": "Abusive",
  "confidence": 94.5,
  "original_text": "you are so stupid",
  "processed_text": "you are so stupid",
  "abusive_prob": 94.5,
  "safe_prob": 5.5
}
```

### `GET /health`

Returns server and model status.

---

## 📊 Dataset

The sample dataset (`dataset.csv`) contains 80 labeled text samples:
- **40 Abusive** sentences (label = 1)
- **40 Not Abusive** sentences (label = 0)

You can extend this dataset with more examples to improve accuracy.

---

## 📈 Model Performance

| Metric     | Value       |
|------------|-------------|
| Model      | Logistic Regression |
| Vectorizer | TF-IDF (1,2-gram)   |
| Accuracy   | ~95%+ on sample set |

> **Note:** Performance improves significantly with a larger, more diverse dataset.

---

## 🔧 Future Improvements

- [ ] Integrate a larger public dataset (e.g., HatEval, Twitter Hate Speech)
- [ ] Add support for multilingual detection
- [ ] Deploy to cloud (Render / Railway / AWS)
- [ ] Add user feedback loop for continuous learning
- [ ] Add LSTM/BERT-based deep learning model

