import streamlit as st
import torch
from transformers import AutoTokenizer


import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader
from torch.optim import AdamW
from transformers import AutoModel, get_linear_schedule_with_warmup
from TorchCRF import CRF
import time

# -----------------------
# 1️⃣ Define DistilBERT + CRF model
# -----------------------
class DistilBertCRF(nn.Module):
    def __init__(self, model_name, num_labels, dropout=0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size # Number of hidden units from DistilBERT. Typically 768 for base models.
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)  # emission scores
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)  # (B, L, H)
        emissions = self.classifier(sequence_output)              # (B, L, num_labels)

        if labels is not None:
            # Create mask for valid tokens
            mask = labels.ne(-100)
            mask[:, 0] = True  # Ensure first timestep is always unmasked

            # Replace masked labels with a safe value 0
            safe_labels = labels.clone()
            safe_labels[~mask] = 0
            safe_labels[:, 0] = safe_labels[:, 0].clamp(0, self.crf.num_tags-1)

            log_likelihood = self.crf(emissions, safe_labels, mask=mask, reduction='mean') # CRF computes sequence-level log-likelihood over the batch.
            return -log_likelihood # loss to minimize for training
        else:
            mask = attention_mask.bool() # Only attend to real tokens (ignore padding)
            mask[:, 0] = True  # first timestep must be on
            return self.crf.decode(emissions, mask=mask)  # Best label sequences, decoding to find most likely label sequence per sentence
            # Output shape: [batch_size, seq_len] (list of predicted labels per token).








# -----------------------------
# Load Model + Tokenizer
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "distilbert-base-multilingual-cased"  # update if different
checkpoint_path = "distilbert/distilbert_crf_ner.pth"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
id2label = {0: 'B-DATE', 1: 'B-LOC', 2: 'B-TIME', 3: 'I-DATE', 4: 'I-LOC', 5: 'I-TIME', 6: 'O'}
num_labels = len(id2label)   # <-- make sure you define id2label = {0:"O", 1:"B-LOC", ...}
model = DistilBertCRF(MODEL_NAME, num_labels).to(device)

# # Load weights
# state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
# model.load_state_dict(state_dict)
# model.eval()

# Load to CPU first, then move model to device
state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
model.load_state_dict(state_dict)
model = model.to(device)   # reassign here
model.eval()

# -----------------------------
# Prediction Function
# -----------------------------
def predict_entities(text):
    # Tokenize while keeping word-piece mapping
    tokens = tokenizer(text, return_tensors="pt", truncation=True, is_split_into_words=False)
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)

    with torch.no_grad():
        preds = model(input_ids, attention_mask=attention_mask)  # CRF decode

    pred_labels = [id2label[i] for i in preds[0]]  # convert ids to tags
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    # Reconstruct words from BPE tokens
    words, labels = [], []
    current_word, current_label = "", None

    for tok, lab in zip(tokens, pred_labels):
        if tok in ["[CLS]", "[SEP]", "[PAD]"]:
            continue

        # Merge subwords (## prefix means continuation)
        if tok.startswith("##"):
            current_word += tok[2:]
        else:
            # If we already built a word, save it
            if current_word:
                words.append(current_word)
                labels.append(current_label)
            current_word = tok
            current_label = lab

    # Add the last word
    if current_word:
        words.append(current_word)
        labels.append(current_label)

    # Now merge entities based on BIO scheme
    entities = []
    current_entity, current_type = "", None

    for word, label in zip(words, labels):
        if label.startswith("B-"):
            # save previous entity if exists
            if current_entity:
                entities.append((current_entity, current_type))
            current_entity = word
            current_type = label[2:]
        elif label.startswith("I-") and current_type == label[2:]:
            current_entity += word
        else:
            if current_entity:
                entities.append((current_entity, current_type))
                current_entity, current_type = "", None

    # Save last one
    if current_entity:
        entities.append((current_entity, current_type))

    return entities

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Burmese Named Entity Recognition (NER) Demo")
st.write("Enter a sentence and the model will highlight entities.")

user_input = st.text_area("Enter your sentence:", "ရန်ကုန်မှာ နေထိုင်ပါတယ်။")

if st.button("Predict"):
    entities = predict_entities(user_input)

    if not entities:
        st.write("No entities found.")
    else:
        st.subheader("Predicted Entities")
        for ent, label in entities:
            st.markdown(f"**{ent}** → `{label}`")
