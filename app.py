"""
Spam Detection System - GUI
Loads the trained models saved from the notebook (tfidf.pkl, models.pkl)
and classifies a message using 5 different ML models.

Run:  python app.py
"""

import os
import re
import tkinter as tk
from tkinter import font as tkfont

import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ================= LOAD MODELS =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

tfidf = joblib.load(os.path.join(BASE_DIR, "tfidf.pkl"))
trained_models = joblib.load(os.path.join(BASE_DIR, "models.pkl"))

# ================= PREPROCESSING =================
# Must be identical to the notebook's preprocessing
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


# ================= COLORS =================
BG = "#0f172a"
PANEL = "#1e293b"
BTN_RUN = "#22c55e"
BTN_CLR = "#ef4444"
FG = "#f8fafc"

IDLE_BG = "#334155"
SPAM_BG = "#dc2626"
HAM_BG = "#16a34a"

MODEL_LIST = list(trained_models.items())


# ================= FUNCTIONS =================
def run_prediction():
    raw = msg_box.get("1.0", tk.END).strip()

    if not raw:
        return

    cleaned = preprocess_text(raw)
    vec = tfidf.transform([cleaned])

    for i, (_, model) in enumerate(MODEL_LIST):
        pred = model.predict(vec)[0]

        if pred == 1:
            res_vars[i].set("⚠ SPAM")
            res_lbls[i].config(bg=SPAM_BG, fg="white")
        else:
            res_vars[i].set("✅ HAM")
            res_lbls[i].config(bg=HAM_BG, fg="white")


def clear_all():
    msg_box.delete("1.0", tk.END)

    for i in range(len(MODEL_LIST)):
        res_vars[i].set("")
        res_lbls[i].config(bg=IDLE_BG, fg=FG)


# ================= WINDOW =================
root = tk.Tk()
root.title("Spam Detection System")
root.geometry("900x800")
root.resizable(True, True)
root.configure(bg=BG)

# ================= FONTS =================
t_title = tkfont.Font(family="Segoe UI", size=22, weight="bold")
t_sub = tkfont.Font(family="Segoe UI", size=11)
t_label = tkfont.Font(family="Segoe UI", size=11)
t_result = tkfont.Font(family="Segoe UI", size=11, weight="bold")
t_btn = tkfont.Font(family="Segoe UI", size=11, weight="bold")

# ================= HEADER =================
tk.Label(
    root,
    text="📧 Spam Detection System",
    font=t_title,
    bg=BG,
    fg="#38bdf8"
).pack(pady=(20, 5))

tk.Label(
    root,
    text="Machine Learning & NLP Project",
    font=t_sub,
    bg=BG,
    fg="#94a3b8"
).pack(pady=(0, 20))

# ================= INPUT PANEL =================
in_panel = tk.Frame(root, bg=PANEL)
in_panel.pack(fill="x", padx=30, pady=10)

tk.Label(
    in_panel,
    text="Enter Message:",
    font=t_label,
    bg=PANEL,
    fg=FG
).pack(anchor="w", padx=15, pady=(12, 5))

msg_box = tk.Text(
    in_panel,
    height=7,
    font=("Segoe UI", 11),
    bg="#020617",
    fg="white",
    insertbackground="white",
    relief="flat",
    padx=10,
    pady=10
)
msg_box.pack(fill="x", padx=15, pady=(0, 15))

# ================= BUTTONS =================
btn_row = tk.Frame(root, bg=BG)
btn_row.pack(pady=15)

tk.Button(
    btn_row,
    text="Process",
    font=t_btn,
    bg=BTN_RUN,
    fg="white",
    relief="flat",
    padx=25,
    pady=10,
    command=run_prediction
).pack(side="left", padx=10)

tk.Button(
    btn_row,
    text="Clear",
    font=t_btn,
    bg=BTN_CLR,
    fg="white",
    relief="flat",
    padx=25,
    pady=10,
    command=clear_all
).pack(side="left", padx=10)

# ================= RESULTS PANEL =================
res_panel = tk.Frame(root, bg=PANEL)
res_panel.pack(fill="both", expand=True, padx=30, pady=15)

tk.Label(
    res_panel,
    text="Classification Results",
    font=("Segoe UI", 13, "bold"),
    bg=PANEL,
    fg="#38bdf8"
).grid(row=0, column=0, columnspan=2, padx=15, pady=15)

res_vars = []
res_lbls = []

for i, (name, _) in enumerate(MODEL_LIST):

    tk.Label(
        res_panel,
        text=name,
        font=t_label,
        bg=PANEL,
        fg=FG,
        width=35,
        anchor="w"
    ).grid(row=i + 1, column=0, padx=20, pady=8, sticky="w")

    var = tk.StringVar()

    lbl = tk.Label(
        res_panel,
        textvariable=var,
        font=t_result,
        bg=IDLE_BG,
        fg=FG,
        width=15,
        pady=5
    )

    lbl.grid(row=i + 1, column=1, padx=20, pady=8)

    res_vars.append(var)
    res_lbls.append(lbl)

root.mainloop()
