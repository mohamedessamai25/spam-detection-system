📧 Spam Detection System

A Machine Learning & NLP project (Spring 2025-2026) that classifies SMS messages as Spam or Ham using five different models, with a Tkinter desktop GUI that shows every model's prediction side by side.

🚀 Features
Text preprocessing: lowercasing, removing URLs/emails/non-letters, stopword removal, Porter stemming
TF-IDF features (15,000 features, unigrams + bigrams)
Five classifiers compared on the same message: Logistic Regression, Random Forest, SVM, Naive Bayes, Decision Tree
Interactive GUI (Tkinter) to test any message
📊 Dataset
SMS Spam Collection (spam.csv): 5,572 messages (13.4% spam)
5,169 messages after removing 403 duplicates
Split: 90% train / 10% test (stratified, random_state=42)
🏆 Results (test set, 517 messages)
Model	Accuracy	Spam Precision	Spam Recall	Spam F1
Logistic Regression	0.9632	0.98	0.72	0.83
Random Forest	0.9381	1.00	0.51	0.67
SVM (linear)	0.9845	0.95	0.92	0.94
Naive Bayes	0.9691	1.00	0.75	0.86
Decision Tree	0.9516	0.90	0.69	0.78

SVM gives the best balance between precision and recall.

🛠️ Installation & Usage
bash
git clone https://github.com/<your-username>/spam-detection-system.git
cd spam-detection-system
python -m pip install -r requirements.txt
python app.py

NLTK stopwords are downloaded automatically on first run.

Important: .pkl files depend on the scikit-learn version. The included models were trained with scikit-learn 1.5.1. If you use a newer version and see an error such as 'SVC' object has no attribute '_effective_probability', either install the pinned version from requirements.txt or regenerate the models:

bash
python train_models.py
📁 Project Structure
├── app.py                        # Tkinter GUI
├── train_models.py               # Re-train & re-save tfidf.pkl / models.pkl
├── models.pkl                    # Trained models
├── tfidf.pkl                     # TF-IDF vectorizer
├── spam.csv                      # Dataset
├── Spam_Detection_System.ipynb   # Full notebook (EDA, training, evaluation)
└── requirements.txt
🧰 Tech Stack

Python, scikit-learn, NLTK, pandas, NumPy, imbalanced-learn, Matplotlib, Seaborn, Tkinter

📄 License

MIT License
