"""Train the fake-news CNN and save it to model/.

Usage:  python train.py            (expects Fake.csv and True.csv in data/ or the project root)

Source giveaways (Reuters datelines, "21st Century Wire", ...) are stripped first so the
model has to learn from the writing itself. See preprocess.py.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score)
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

from preprocess import strip_source_leakage

SEED = 42
MAX_TOKENS = 20000
SEQ_LEN = 400
MODEL_DIR = Path("model")
DOCS_DIR = Path("docs")

def find_csv(name):
    for folder in (Path("data"), Path(".")):
        if (folder / name).exists():
            return folder / name
    raise SystemExit(f"{name} not found. Download it from Kaggle (see README) into data/.")


def load_data():
    fake = pd.read_csv(find_csv("Fake.csv")).assign(label=0)
    real = pd.read_csv(find_csv("True.csv")).assign(label=1)
    df = pd.concat([fake, real], ignore_index=True)
    df["raw"] = (df["title"].fillna("") + ". " + df["text"].fillna("")).str.strip()
    df["clean"] = df["raw"].map(strip_source_leakage)
    df = df[df["clean"].str.split().str.len() >= 5]
    return df.drop_duplicates(subset="clean").reset_index(drop=True)


def tfidf_baseline(train_text, test_text, y_train, y_test):
    vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
    clf = LogisticRegression(max_iter=2000).fit(vec.fit_transform(train_text), y_train)
    return accuracy_score(y_test, clf.predict(vec.transform(test_text)))


def build_cnn(train_text):
    vectorize = layers.TextVectorization(max_tokens=MAX_TOKENS, output_sequence_length=SEQ_LEN)
    vectorize.adapt(tf.data.Dataset.from_tensor_slices(train_text).batch(256))
    model = keras.Sequential([
        keras.Input(shape=(1,), dtype=tf.string),
        vectorize,
        layers.Embedding(MAX_TOKENS, 100),
        layers.Conv1D(128, 5, activation="relu"),
        layers.GlobalMaxPooling1D(),
        layers.Dropout(0.5),
        layers.Dense(64, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def main():
    keras.utils.set_random_seed(SEED)
    df = load_data()
    reuters_share = df.loc[df.label == 1, "raw"].str.contains("reuters", case=False).mean()
    print(f"{len(df)} unique articles ({df.label.mean():.1%} real); "
          f"{reuters_share:.1%} of real articles mention 'Reuters'")

    train, test = train_test_split(df, test_size=0.15, stratify=df.label, random_state=SEED)
    train, val = train_test_split(train, test_size=0.15, stratify=train.label, random_state=SEED)

    # How much of the "99%" comes from the source giveaways? Same model, raw vs cleaned text.
    leaky_acc = tfidf_baseline(train.raw, test.raw, train.label, test.label)
    clean_acc = tfidf_baseline(train.clean, test.clean, train.label, test.label)
    print(f"TF-IDF + LogReg: raw text {leaky_acc:.4f} | leakage removed {clean_acc:.4f}")

    model = build_cnn(train.clean.to_numpy())
    model.fit(train.clean.to_numpy(), train.label.to_numpy(),
              validation_data=(val.clean.to_numpy(), val.label.to_numpy()),
              epochs=10, batch_size=64, verbose=2,
              callbacks=[keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)])

    prob = model.predict(test.clean.to_numpy(), verbose=0).ravel()
    pred = (prob >= 0.5).astype(int)
    print(classification_report(test.label, pred, target_names=["Fake", "Real"], digits=4))

    metrics = {
        "cnn_test_accuracy": round(accuracy_score(test.label, pred), 4),
        "cnn_test_f1": round(f1_score(test.label, pred), 4),
        "tfidf_logreg_raw_accuracy": round(leaky_acc, 4),
        "tfidf_logreg_clean_accuracy": round(clean_acc, 4),
        "real_articles_mentioning_reuters": round(float(reuters_share), 4),
        "train_size": len(train), "val_size": len(val), "test_size": len(test),
    }
    print(json.dumps(metrics, indent=2))

    DOCS_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_predictions(test.label, pred, display_labels=["Fake", "Real"],
                                            cmap="Blues", colorbar=False, ax=ax)
    ax.set_title(f"CNN test set (accuracy {metrics['cnn_test_accuracy']:.1%})")
    fig.tight_layout()
    fig.savefig(DOCS_DIR / "confusion_matrix.png", dpi=110)

    MODEL_DIR.mkdir(exist_ok=True)
    model.save(MODEL_DIR / "fake_news_cnn.keras")
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved model to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
