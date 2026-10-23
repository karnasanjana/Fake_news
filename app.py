"""Streamlit app: paste a news article, get a Real / Fake prediction from the CNN.

Run:  streamlit run app.py
"""
import json
from pathlib import Path

import numpy as np
import streamlit as st
from tensorflow import keras

from preprocess import strip_source_leakage

MODEL_DIR = Path(__file__).parent / "model"

EXAMPLES = {
    "Wire-service style": (
        "Poland's new prime minister faces a difficult balancing act as he seeks to repair "
        "relations with the European Union while keeping the support of the ruling party's "
        "conservative base, officials said on Tuesday."
    ),
    "Sensational style": (
        "Donald Trump Sends Out Embarrassing New Year's Eve Message; This is Disturbing. "
        "You won't believe what he tweeted this time. Sharing is caring, so make sure "
        "everyone sees just how unhinged this man has become!"
    ),
}


@st.cache_resource
def load_model():
    return keras.models.load_model(MODEL_DIR / "fake_news_cnn.keras")


@st.cache_data
def load_metrics():
    path = MODEL_DIR / "metrics.json"
    return json.loads(path.read_text()) if path.exists() else {}


st.set_page_config(page_title="Fake News Detector", page_icon="📰")
st.title("📰 Fake News Detector")
st.write(
    "A 1D-CNN trained on ~40k political news articles (ISOT dataset) classifies an article "
    "as **Real** or **Fake** from its writing style and content."
)

metrics = load_metrics()
if metrics:
    c1, c2, c3 = st.columns(3)
    c1.metric("Test accuracy", f"{metrics['cnn_test_accuracy']:.1%}")
    c2.metric("Test F1", f"{metrics['cnn_test_f1']:.3f}")
    c3.metric("Test articles", f"{metrics['test_size']:,}")

example = st.selectbox("Try an example, or paste your own below", ["(none)", *EXAMPLES])
text = st.text_area("News article (title and/or body)",
                    value=EXAMPLES.get(example, ""), height=220)

if st.button("Predict", type="primary"):
    cleaned = strip_source_leakage(text)
    if len(cleaned.split()) < 5:
        st.error("Please enter at least a sentence or two of the article.")
    else:
        prob_real = float(load_model().predict(np.array([cleaned], dtype=object), verbose=0)[0][0])
        label, conf = ("Real", prob_real) if prob_real >= 0.5 else ("Fake", 1 - prob_real)
        (st.success if label == "Real" else st.error)(f"**{label}** ({conf:.1%} confidence)")
        st.progress(prob_real, text=f"Probability real: {prob_real:.1%}")

st.caption(
    "Trained on 2016–2017 US political news, so it works best on similar articles. It judges "
    "style and content, not facts, so it is not a fact-checker."
)
