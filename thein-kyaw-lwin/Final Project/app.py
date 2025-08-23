
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import json
from transformers import AutoTokenizer, AutoModel
import plotly.graph_objects as go
from math import radians, sin, cos, asin, sqrt

# ---------------------------
# Model Architectures (exactly two distinct heads)
# ---------------------------
class ImprovedMyanmarBertGeocoder(nn.Module):
    """
    BERT-base-multilingual-cased backbone
    Regression head: 768 -> 512 -> 256 -> 128 -> 2 + Sigmoid
    Uses pooler_output
    """
    def __init__(self, model_name="bert-base-multilingual-cased", dropout_rate=0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        bert_hidden_size = self.bert.config.hidden_size  # 768
        self.dropout = nn.Dropout(dropout_rate)
        self.coordinate_regressor = nn.Sequential(
            nn.Linear(bert_hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(128, 2),
            nn.Sigmoid()
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.coordinate_regressor:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        cls_output = outputs.pooler_output  # [batch, 768]
        cls_output = self.dropout(cls_output)
        return self.coordinate_regressor(cls_output)


class ImprovedMyanmarDistilBertGeocoder(nn.Module):
    """
    DistilBERT-base-multilingual-cased backbone
    Regression head: 768 -> 384 -> 192 -> 96 -> 2 + Sigmoid
    Uses [CLS] embedding from last_hidden_state[:, 0]
    """
    def __init__(self, model_name="distilbert-base-multilingual-cased", dropout_rate=0.3):
        super().__init__()
        self.distilbert = AutoModel.from_pretrained(model_name)
        distilbert_hidden_size = self.distilbert.config.hidden_size  # 768
        self.dropout = nn.Dropout(dropout_rate)
        self.coordinate_regressor = nn.Sequential(
            nn.Linear(distilbert_hidden_size, 384),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(384, 192),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(192, 96),
            nn.ReLU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(96, 2),
            nn.Sigmoid()
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.coordinate_regressor:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        cls_output = outputs.last_hidden_state[:, 0]  # [batch, 768]
        cls_output = self.dropout(cls_output)
        return self.coordinate_regressor(cls_output)


# ---------------------------
# Utils
# ---------------------------
@st.cache_resource
def load_resources():
    """Load both models, tokenizers, and preprocessing scalers once."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tokenizers
    bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    distil_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")

    # Models (init with correct backbones)
    bert_model = ImprovedMyanmarBertGeocoder("bert-base-multilingual-cased").to(device)
    distil_model = ImprovedMyanmarDistilBertGeocoder("distilbert-base-multilingual-cased").to(device)

    # Load checkpoints saved as {'model_state_dict': ...}
    try:
        ckpt = torch.load("bert_model.pt", map_location=device)
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        bert_model.load_state_dict(state_dict)
        bert_loaded = True
    except Exception as e:
        st.warning(f"Could not load bert_model.pt: {e}")
        bert_model = None
        bert_loaded = False

    try:
        ckpt = torch.load("distilbert_model.pt", map_location=device)
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        distil_model.load_state_dict(state_dict)
        distil_loaded = True
    except Exception as e:
        st.warning(f"Could not load distilbert_model.pt: {e}")
        distil_model = None
        distil_loaded = False

    # Preprocessing scalers
    bert_scalers = None
    distil_scalers = None
    try:
        with open("bert_preprocessing_info.json", "r") as f:
            bert_preproc = json.load(f)
        bert_scalers = (
            {"min": bert_preproc["lat_scaler_min"], "scale": bert_preproc["lat_scaler_scale"]},
            {"min": bert_preproc["lon_scaler_min"], "scale": bert_preproc["lon_scaler_scale"]},
        )
    except Exception as e:
        st.warning(f"Could not load bert_preprocessing_info.json: {e}")

    try:
        with open("distilbert_preprocessing_info.json", "r") as f:
            distil_preproc = json.load(f)
        distil_scalers = (
            {"min": distil_preproc["lat_scaler_min"], "scale": distil_preproc["lat_scaler_scale"]},
            {"min": distil_preproc["lon_scaler_min"], "scale": distil_preproc["lon_scaler_scale"]},
        )
    except Exception as e:
        st.warning(f"Could not load distilbert_preprocessing_info.json: {e}")

    return {
        "device": device,
        "bert": {"model": bert_model, "tokenizer": bert_tokenizer, "scalers": bert_scalers, "loaded": bert_loaded},
        "distil": {"model": distil_model, "tokenizer": distil_tokenizer, "scalers": distil_scalers, "loaded": distil_loaded},
    }


def denormalize_coordinates(normalized_coords: np.ndarray, lat_scaler_info, lon_scaler_info):
    lat_min, lat_scale = lat_scaler_info["min"], lat_scaler_info["scale"]
    lon_min, lon_scale = lon_scaler_info["min"], lon_scaler_info["scale"]
    latitude = (normalized_coords[:, 0] / lat_scale) + lat_min
    longitude = (normalized_coords[:, 1] / lon_scale) + lon_min
    return latitude, longitude


def haversine_km(lat1, lon1, lat2, lon2):
    # convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    km = 6371.0088 * c
    return km


def predict_for_model(address: str, bundle: dict):
    model = bundle["model"]
    tokenizer = bundle["tokenizer"]
    scalers = bundle["scalers"]
    device = load["device"]
    if model is None or scalers is None:
        return None

    model.eval()
    with torch.no_grad():
        inputs = tokenizer(
            address,
            return_tensors="pt",
            max_length=128,
            padding="max_length",
            truncation=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        normalized = model(inputs["input_ids"], inputs["attention_mask"]).cpu().numpy()
        lat, lon = denormalize_coordinates(normalized, scalers[0], scalers[1])
        return float(lat[0]), float(lon[0])


# ---------------------------
# UI
# ---------------------------
st.set_page_config(page_title="Myanmar Geocoding: BERT vs DistilBERT", layout="centered")
st.title("📍 Myanmar Address → Coordinate Prediction")
st.caption("Two distinct architectures: BERT vs DistilBERT. Enter an address and ground-truth coordinate.")

# Inputs
address = st.text_area("Address", placeholder="Enter Myanmar address text here...", height=80)
c1, c2 = st.columns(2)
with c1:
    gt_lat = st.number_input("Ground Truth Latitude", format="%.6f", value=0.0)
with c2:
    gt_lon = st.number_input("Ground Truth Longitude", format="%.6f", value=0.0)

load = load_resources()

if st.button("Predict with Both Models", type="primary", use_container_width=True):
    if not address.strip():
        st.error("Please enter an address.")
    else:
        bert_pred = predict_for_model(address, load["bert"]) if load["bert"]["loaded"] else None
        distil_pred = predict_for_model(address, load["distil"]) if load["distil"]["loaded"] else None

        # Results table
        rows = []
        if bert_pred:
            rows.append(("BERT", bert_pred[0], bert_pred[1], haversine_km(gt_lat, gt_lon, bert_pred[0], bert_pred[1])))
        if distil_pred:
            rows.append(("DistilBERT", distil_pred[0], distil_pred[1], haversine_km(gt_lat, gt_lon, distil_pred[0], distil_pred[1])))

        if not rows:
            st.error("No predictions available. Ensure model files and preprocessing JSONs are present.")
        else:
            st.subheader("Predicted Coordinates")
            for name, lat, lon, err in rows:
                st.write(f"**{name}** → Lat: `{lat:.6f}`, Lon: `{lon:.6f}`  •  Error vs GT: `{err:.2f} km`")

            # Plotly map
            fig = go.Figure()

            # Ground truth marker
            fig.add_trace(go.Scattermap(
                lat=[gt_lat], lon=[gt_lon],
                mode="markers+text",
                marker=dict(size=14),
                text=["Ground Truth"],
                textposition="top right",
                name="Ground Truth"
            ))

            # BERT marker
            if bert_pred:
                fig.add_trace(go.Scattermap(
                    lat=[bert_pred[0]], lon=[bert_pred[1]],
                    mode="markers+text",
                    marker=dict(size=12),
                    text=["BERT"],
                    textposition="top right",
                    name="BERT"
                ))

            # DistilBERT marker
            if distil_pred:
                fig.add_trace(go.Scattermap(
                    lat=[distil_pred[0]], lon=[distil_pred[1]],
                    mode="markers+text",
                    marker=dict(size=12),
                    text=["DistilBERT"],
                    textposition="top right",
                    name="DistilBERT"
                ))

            # Center map on ground truth (fallback to first pred if GT is 0,0)
            center_lat, center_lon = gt_lat, gt_lon
            if center_lat == 0.0 and center_lon == 0.0:
                if bert_pred: center_lat, center_lon = bert_pred
                elif distil_pred: center_lat, center_lon = distil_pred

            fig.update_layout(
                mapbox_style="open-street-map",
                mapbox=dict(center={"lat": center_lat, "lon": center_lon}, zoom=12),
                margin={"r":0, "t":0, "l":0, "b":0},
                legend=dict(orientation="h", yanchor="bottom", y=0.01)
            )
            st.plotly_chart(fig, use_container_width=True)

            st.success("Done!")

st.markdown("""
**Files expected in the same directory:**
- `bert_model.pt`, `distilbert_model.pt`
- `bert_preprocessing_info.json`, `distilbert_preprocessing_info.json`
""")
