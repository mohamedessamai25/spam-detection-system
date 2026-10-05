"""
Re-train the 5 models and re-save tfidf.pkl / models.pkl
using the scikit-learn version installed on THIS machine.
Same preprocessing and settings as Spam_Detection_System.ipynb.

Run:  python train_models.py
"""
import os
import re

import joblib
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

nltk.download("stopwords", quiet=True)
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))


def preprocess_text(text):
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [stemmer.stem(w) for w in text.split()
              if w not in stop_words and len(w) > 1]
    return " ".join(tokens)


df = pd.read_csv(os.path.join(BASE_DIR, "spam.csv"), encoding="latin-1")
df = df[["v1", "v2"]]
df.columns = ["label", "text"]
df["label_num"] = (df["label"] == "spam").astype(int)
df = df.drop_duplicates().reset_index(drop=True)
df["cleaned_text"] = df["text"].apply(preprocess_text)

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    df["cleaned_text"], df["label_num"],
    test_size=0.1, random_state=42, stratify=df["label_num"]
)

tfidf = TfidfVectorizer(max_features=15000, ngram_range=(1, 2))
X_train = tfidf.fit_transform(X_train_raw)
X_test = tfidf.transform(X_test_raw)

models = {
    "Logistic Regression": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=20,
                                            min_samples_split=5, min_samples_leaf=2,
                                            random_state=42, n_jobs=-1),
    "SVM": SVC(kernel="linear", class_weight="balanced",
               probability=True, random_state=42),
    "Naive Bayes": MultinomialNB(alpha=1.0),
    "Decision Tree": DecisionTreeClassifier(max_depth=20, min_samples_split=5,
                                            min_samples_leaf=2, random_state=42),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"{name:20s} test accuracy: {acc:.4f}")

joblib.dump(tfidf, os.path.join(BASE_DIR, "tfidf.pkl"))
joblib.dump(models, os.path.join(BASE_DIR, "models.pkl"))
print("Saved tfidf.pkl and models.pkl")
