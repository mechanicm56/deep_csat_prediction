import streamlit as st
import torch
import pickle
import gdown
from pathlib import Path
import numpy as np
import tensorflow as tfa
import gdown, zipfile, os
from transformers import BertTokenizer, BertForSequenceClassification
from tensorflow.keras.preprocessing.sequence import pad_sequences


# --------------------------------------------------
# GOOGLE DRIVE FILE IDS
# --------------------------------------------------
FILE_IDS = {
    "lstm": "1PLBJYlIshDk1Br9vNptfXHJFPRNHDO8i",
    "gru": "1SndjABYlVadUQ7-9XKm4HR1Mf7Mbxir8",
    "bert": "1P5cm9nfEfoYD6dnb9V48dobtaQCORnq0",
}

# Where to store downloaded files locally
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# --------------------------------------------------
# HELPER: Download file from Drive if not exists
# --------------------------------------------------
def download_from_drive(file_id, output_path):
    if not output_path.exists():
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, str(output_path), quiet=False)
    return output_path


# --------------------------------------------------
# LOAD MODELS & DATA
# --------------------------------------------------
@st.cache_resource
def load_models():
    lstm_path = download_from_drive(FILE_IDS["kmeans"], MODEL_DIR / "lstm_csat_model.keras")
    gru_path = download_from_drive(FILE_IDS["scaler"], MODEL_DIR / "gru_csat_model.keras")
    bert_path = download_from_drive(FILE_IDS["similarity"], MODEL_DIR / "distilbert_csat_model.pth")
    # products_path = download_from_drive(FILE_IDS["products"], MODEL_DIR / "product_names.pkl")

    # Load with joblib
    lstm_model = tf.keras.models.load_model(lstm_path)
    gru_model = tf.keras.models.load_model(gru_path)
    bert_model = torch.load(bert_path, map_location=torch.device("cpu"))
    # products = joblib.load(products_path)

    return lstm_model, gru_model, bert_model

@st.cache_resource
def load_tokenizer():

    TOKENIZER_ID = "15fXz4-w5ykPqe7y0GAH2W-aNSeARpGXE"
    TOKENIZER_ZIP = MODEL_DIR / "bert_tokenizer.zip"
    TOKENIZER_DIR = MODEL_DIR / "bert_tokenizer"

    if not os.path.exists(TOKENIZER_DIR):

        gdown.download(
            f"https://drive.google.com/uc?id={TOKENIZER_ID}",
            TOKENIZER_ZIP,
            quiet=False
        )

        with zipfile.ZipFile(TOKENIZER_ZIP, 'r') as zip_ref:
            zip_ref.extractall(TOKENIZER_DIR)

    tokenizer = BertTokenizer.from_pretrained(TOKENIZER_DIR)

    return tokenizer

lstm_model, gru_model, bert_model = load_models()

# Page title
st.title("Customer Satisfaction (CSAT) Prediction System")

st.write("Predict CSAT score from customer remarks using Deep Learning models.")

# ------------------------------
# Load LSTM Model
# ------------------------------

# lstm_model = tf.keras.models.load_model("lstm_csat_model.h5")

with open("tokenizer.pkl", "rb") as f:
    lstm_tokenizer = pickle.load(f)

MAX_LEN = 100

# ------------------------------
# Load BERT Model
# ------------------------------

bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

bert_model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=1
)

bert_model.load_state_dict(bert_model)
bert_model.eval()

# ------------------------------
# Model Selection
# ------------------------------

model_choice = st.selectbox(
    "Select Model for Prediction",
    ["LSTM Model (TensorFlow)", "BERT Model (PyTorch)"]
)

# ------------------------------
# User Input
# ------------------------------

customer_remark = st.text_area("Enter Customer Remark")

# ------------------------------
# Prediction Button
# ------------------------------

if st.button("Predict CSAT Score"):

    if customer_remark.strip() == "":
        st.warning("Please enter customer remark")

    else:

        # ------------------------------
        # LSTM Prediction
        # ------------------------------
        if model_choice == "LSTM Model (TensorFlow)":

            seq = lstm_tokenizer.texts_to_sequences([customer_remark])
            padded = pad_sequences(seq, maxlen=MAX_LEN)

            prediction = lstm_model.predict(padded)

            csat_score = float(prediction[0][0])

        # ------------------------------
        # BERT Prediction
        # ------------------------------
        else:

            inputs = bert_tokenizer(
                customer_remark,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            )

            with torch.no_grad():
                outputs = bert_model(**inputs)

            csat_score = outputs.logits.item()

        # ------------------------------
        # Display Prediction
        # ------------------------------

        st.subheader("Predicted CSAT Score")
        st.success(round(csat_score, 2))

        # Satisfaction Category
        if csat_score <= 2:
            st.error("Customer is Dissatisfied")

        elif csat_score <= 3:
            st.warning("Customer is Neutral")

        else:
            st.success("Customer is Satisfied")