# Burmese News Classification — Streamlit UI

This folder contains a ready-to-run **Streamlit** app for your Burmese news classifier.

## 1) Project layout

```
your_project/
├─ app.py
├─ models/
│  ├─ bilstm_mynews.keras
│  ├─ nb_tfidf.joblib
│  ├─ keras_tokenizer.pkl
│  ├─ label_encoder.pkl
│  ├─ config.json              # should include {"max_len": <int>} (or similar)
│  └─ stopwords.txt            # optional
└─ requirements.txt
```

## 2) Setup

### Windows (Python 3.10 recommended)

```powershell
# from project root
py -3.10 -m venv .venv
.venv\Scripts\activate

py -3.10 -m pip install --upgrade pip
py -3.10 -m pip install -r requirements.txt
```

If TensorFlow/protobuf prints version warnings, pin protobuf to `3.20.*`:

```powershell
py -3.10 -m pip install "protobuf==3.20.*" --force-reinstall
```

## 3) Run

```powershell
python -m streamlit run app.py
# or
py -3.10 -m streamlit run app.py
```

Then open the local URL shown in the terminal (usually http://localhost:8501).

## 4) Notes

- **BiLSTM model**: the app reads `config.json` for `max_len`. If missing, it defaults to 200.
- **Tokenizer**: must match the one used during training (`keras_tokenizer.pkl`).
- **Label encoder**: used to convert predicted indices back to class names.
- **Stopwords**: optional. If present, they are applied during preprocessing.
- **Burmese tokenization**: the app will use **PyIDAUNGSU** if available; otherwise it falls back
  to a simple regex-based tokenizer so it still works.

## 5) Deploy (optional: Streamlit Community Cloud)

1. Push your repo to GitHub.
2. On Streamlit Community Cloud, create a new app, point it to `app.py`.
3. Set Python version to 3.10 in the advanced settings, and make sure your `requirements.txt` is in the repo.