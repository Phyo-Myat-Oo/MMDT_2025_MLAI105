# Streamlit UI for Burmese News Classification
# -------------------------------------------------
# Run locally:
#   py -3.10 -m streamlit run app.py
#
import os, json, pickle, unicodedata, re, hashlib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import pyidaungsu as pds
import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------------- Paths ----------------
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
PATHS = {
    "bilstm":        os.path.join(MODELS_DIR, "bilstm_mynews.keras"),
    "nb":            os.path.join(MODELS_DIR, "nb_tfidf.joblib"),
    "tokenizer":     os.path.join(MODELS_DIR, "keras_tokenizer.pkl"),
    "label_encoder": os.path.join(MODELS_DIR, "label_encoder.pkl"),
    "config":        os.path.join(MODELS_DIR, "config.json"),
    "stopwords":     os.path.join(MODELS_DIR, "stopwords.txt"),
}

# ---------------- Helpers ----------------
def _read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _load_stopwords(path):
    sw = set()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    sw.add(line)
    return sw

def strip_control(text: str) -> str:
    return "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")

BURMESE_BLOCK = r"\u1000-\u109F"
TOK_RGX = re.compile(fr"[{BURMESE_BLOCK}]+|[A-Za-z]+|\d+|[^\s\w]", re.UNICODE)

def naive_tokenize(text: str):
    return TOK_RGX.findall(strip_control(text))

def tokenize(text: str):
    if pds is None:
        return naive_tokenize(text)
    try:
        if hasattr(pds, "tokenize"):
            toks = pds.tokenize(text)
            if isinstance(toks, list) and toks and isinstance(toks[0], list):
                toks = toks[0]
            return [t for t in toks if t.strip()]
    except Exception:
        pass
    try:
        if hasattr(pds, "segment"):
            toks = pds.segment(text)
            return [t for t in toks if t.strip()]
    except Exception:
        pass
    return naive_tokenize(text)

def build_preprocess():
    sw = _load_stopwords(PATHS["stopwords"])
    def preprocess(t: str) -> str:
        if not t:
            return ""
        toks = tokenize(t)
        out = []
        for tok in toks:
            tok = unicodedata.normalize("NFKC", tok).strip()
            if not tok or tok in sw:
                continue
            out.append(tok.lower())
        return " ".join(out)
    return preprocess

# --- Tokenizer shims for joblib  ---
_STOPWORDS_CACHE = None
def _get_sw():
    global _STOPWORDS_CACHE
    if _STOPWORDS_CACHE is None:
        _STOPWORDS_CACHE = _load_stopwords(PATHS["stopwords"])
    return _STOPWORDS_CACHE

def _normalize_token(tok: str) -> str:
    tok = unicodedata.normalize("NFKC", tok).strip()
    return tok.lower()

WS = re.compile(r"[\s\u00A0]+")
def _norm(t): 
    return unicodedata.normalize("NFC", t or "")
def _clean(t): 
    return WS.sub(" ", _norm(t)).strip()

def tokenize_burmese(t):
    toks = pds.tokenize(_clean(t), form="word")
    return [x.lower().strip() for x in toks if x.strip()]

def tokenize_burmese_stopwords(t):
    return tokenize_burmese(t)

# ---------------- Loaders ----------------
@st.cache_resource(show_spinner=False)
def load_label_encoder():
    with open(PATHS["label_encoder"], "rb") as f:
        return pickle.load(f)

@st.cache_resource(show_spinner=False)
def load_nb():
    print(PATHS["nb"])
    return joblib.load(PATHS["nb"])

@st.cache_resource(show_spinner=False)
def load_bilstm_bundle():
    if tf is None:
        raise RuntimeError("TensorFlow not installed")

    model = tf.keras.models.load_model(PATHS["bilstm"], custom_objects={})

    inp_shape = model.input_shape
    if isinstance(inp_shape, (list, tuple)) and isinstance(inp_shape[0], (list, tuple)):
        inp_shape = inp_shape[0]
    expected_len = None
    if isinstance(inp_shape, (list, tuple)) and len(inp_shape) >= 2:
        if inp_shape[1] is not None:
            expected_len = int(inp_shape[1])

    with open(PATHS["tokenizer"], "rb") as f:
        tokenizer = pickle.load(f)
    cfg = _read_json(PATHS["config"])
    cfg_len = int(cfg.get("max_len") or cfg.get("maxlen") or 350)
    max_len = expected_len or cfg_len

    return model, tokenizer, max_len

# ---------------- Predict ----------------
def predict_nb(text: str):
    pipe = load_nb()
    print(text)
    print(pipe)
    proba = pipe.predict_proba([text])[0]
    print(proba)
    return np.array(proba, dtype=float), pipe.classes_

def predict_bilstm(text: str):
    toks = " ".join(tokenize_burmese(text))

    model, tokenizer, max_len = load_bilstm_bundle()
    le = load_label_encoder()

    seq = tokenizer.texts_to_sequences([toks])
    X_pad = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post", dtype="int32")

    proba = model.predict(X_pad, verbose=0)[0]
    return np.array(proba, dtype=float), le.classes_

# ---------------- UI ----------------
st.set_page_config(page_title="Burmese News Classifier", page_icon="📰", layout="centered")
st.title("📰 Burmese News Classifier")

with st.sidebar:
    model_choice = st.selectbox("Model", ["BiLSTM (Keras)", "Naive Bayes (TF-IDF)"], index=0)
    show_debug = st.toggle("Show debug panel", value=False)

text = st.text_area("Burmese text", height=160, placeholder="Paste a Burmese headline/paragraph…")

if st.button("Classify"):
    t = (text or "").strip()
    if not t:
        st.warning("Please enter some Burmese text.")
        st.stop()

    try:
        if model_choice.startswith("BiLSTM"):
            proba, labels = predict_bilstm(t)
        else:
            proba, labels = predict_nb(t)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    order = np.argsort(proba)[::-1]
    top = order[0]
    st.subheader("Prediction")
    st.write(f"**Top class:** `{labels[top]}`")
    st.write(f"**Score:** `{proba[top]:.4f}`")

    df = pd.DataFrame({"class": labels, "probability": proba}).sort_values(
        "probability", ascending=False
    ).reset_index(drop=True)
    st.dataframe(df, use_container_width=True)

    if show_debug:
        with st.expander("Debug — probability vector", expanded=False):
            st.json([{ "label": labels[i], "p": float(proba[i]) } for i in range(len(labels))])
