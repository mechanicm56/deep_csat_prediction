import streamlit as st
import tensorflow as tf
import torch
import gdown
import os
import zipfile
from pathlib import Path
from transformers import BertTokenizer, BertForSequenceClassification
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------
# CONFIGURATION
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent

print("Base DIR", BASE_DIR)

LSTM_MODEL_ID = "1PLBJYlIshDk1Br9vNptfXHJFPRNHDO8i"
GRU_MODEL_ID = "1SndjABYlVadUQ7-9XKm4HR1Mf7Mbxir8"
BERT_MODEL_ID = "1P5cm9nfEfoYD6dnb9V48dobtaQCORnq0"
TOKENIZER_ID = "1SVVnDOA1ABGVFURVDR3RXq3GQQD4LJQ5"

MAX_LEN = 100

# Where to store downloaded files locally
MODEL_DIR = Path(BASE_DIR / "models")
MODEL_DIR.mkdir(exist_ok=True)

# -----------------------------
# DOWNLOAD FUNCTION
# -----------------------------

def download_file(file_id, output):

    if not os.path.exists(output):

        url = f"https://drive.google.com/uc?id={file_id}"

        gdown.download(url, output, quiet=False)


# -----------------------------
# LOAD LSTM MODEL
# -----------------------------

@st.cache_resource
def load_lstm():

    download_file(LSTM_MODEL_ID, MODEL_DIR / "lstm_model.keras")

    model = tf.keras.models.load_model(MODEL_DIR / "lstm_model.keras")

    return model


# -----------------------------
# LOAD GRU MODEL
# -----------------------------

@st.cache_resource
def load_gru():

    download_file(GRU_MODEL_ID, MODEL_DIR / "gru_model.keras")

    model = tf.keras.models.load_model(MODEL_DIR / "gru_model.keras")

    return model


# -----------------------------
# LOAD BERT MODEL
# -----------------------------

@st.cache_resource
def load_bert():

    download_file(BERT_MODEL_ID, MODEL_DIR / "distilbert_model.pth")

    model = BertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=1
    )

    model.load_state_dict(
        torch.load(MODEL_DIR / "distilbert_model.pth", map_location=torch.device("cpu"))
    )

    model.eval()

    return model


# -----------------------------
# LOAD TOKENIZER
# -----------------------------

@st.cache_resource
def load_tokenizer():

    download_file(TOKENIZER_ID, MODEL_DIR / "bert_tokenizer.zip")

    if not os.path.exists(MODEL_DIR / "bert_tokenizer"):

        with zipfile.ZipFile("bert_tokenizer.zip", "r") as zip_ref:

            zip_ref.extractall("bert_tokenizer")

    tokenizer = BertTokenizer.from_pretrained("bert_tokenizer")

    return tokenizer


# -----------------------------
# LOAD EVERYTHING
# -----------------------------

lstm_model = load_lstm()
gru_model = load_gru()
bert_model = load_bert()
bert_tokenizer = load_tokenizer()

# -----------------------------
# STREAMLIT UI
# -----------------------------

st.title("Customer Satisfaction (CSAT) Prediction System")

st.write("Predict customer satisfaction score from remarks using deep learning models.")

model_choice = st.selectbox(
    "Select Model",
    ["LSTM", "GRU", "BERT"]
)

customer_remark = st.text_area("Enter Customer Remark")

# -----------------------------
# PREDICTION
# -----------------------------

if st.button("Predict CSAT Score"):

    if customer_remark.strip() == "":

        st.warning("Please enter customer remark")

    else:

        # -------------------------
        # LSTM MODEL
        # -------------------------
        if model_choice == "LSTM":

            sequences = bert_tokenizer.encode(
                customer_remark,
                truncation=True,
                padding="max_length",
                max_length=MAX_LEN
            )

            padded = [sequences]

            prediction = lstm_model.predict(padded)

            csat_score = float(prediction[0][0])

        # -------------------------
        # GRU MODEL
        # -------------------------
        elif model_choice == "GRU":

            sequences = bert_tokenizer.encode(
                customer_remark,
                truncation=True,
                padding="max_length",
                max_length=MAX_LEN
            )

            padded = [sequences]

            prediction = gru_model.predict(padded)

            csat_score = float(prediction[0][0])

        # -------------------------
        # BERT MODEL
        # -------------------------
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

        # -----------------------------
        # OUTPUT
        # -----------------------------

        st.subheader("Predicted CSAT Score")

        st.success(round(csat_score, 2))

        if csat_score <= 2:

            st.error("Customer Dissatisfied")

        elif csat_score <= 3:

            st.warning("Customer Neutral")

        else:

            st.success("Customer Satisfied")