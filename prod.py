import streamlit as st
import numpy as np
import pickle
import pandas as pd
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


BASE_DIR = Path(__file__).resolve().parent

# Where to store downloaded files locally
MODEL_DIR = Path(BASE_DIR / "models")

# -----------------------------
# PAGE CONFIG
# -----------------------------

st.set_page_config(
    page_title="CSAT Prediction Dashboard",
    layout="wide"
)

st.title("📊 Customer Satisfaction Prediction Dashboard")


# -----------------------------
# LOAD TOKENIZER + SCALER
# -----------------------------

@st.cache_resource
def load_resources():

    tokenizer = pickle.load(open(MODEL_DIR / "lstm/tokenizer.pkl","rb"))
    scaler = pickle.load(open(MODEL_DIR / "lstm/scaler.pkl","rb"))

    return tokenizer, scaler


tokenizer, scaler = load_resources()


# -----------------------------
# MODEL LOADER
# -----------------------------

@st.cache_resource
def load_model_selected(model_name):

    if model_name == "LSTM":
        return load_model(MODEL_DIR / "lstm/lstm_model.keras")

    else:
        return load_model(MODEL_DIR / "bilstm/bilstm_model.keras")

max_len = 100

# -----------------------------
# INPUT GRID
# -----------------------------

col1, col2, col3 = st.columns(3)

with col1:

    channel = st.selectbox(
        "Channel Name",
        ["Chat","Email","Phone","Social Media"]
    )

    category = st.selectbox(
        "Category",
        ["Delivery","Product Issue","Return","Payment"]
    )

    subcategory_options = {
        "Delivery": ["Late Delivery"],
        "Product Issue": ["Damaged Item"],
        "Return": ["Wrong Item"],
        "Payment": ["Refund Issue"]
    }

    sub_category = st.selectbox(
        "Sub Category",
        subcategory_options[category]
    )

with col2:

    city = st.selectbox(
        "Customer City",
        ["Delhi","Mumbai","Bangalore","Chennai"]
    )

    product_category = st.selectbox(
        "Product Category",
        ["Electronics","Fashion","Home","Beauty"]
    )

    tenure = st.selectbox(
        "Agent Tenure",
        ["0-6 months","6-12 months","1-2 years"]
    )

with col3:

    shift = st.selectbox(
        "Agent Shift",
        ["Morning","Evening","Night"]
    )

    item_price = st.number_input("Item Price", min_value=0.0)

    handling_time = st.number_input("Handling Time", min_value=0.0)


# Second row

col4, col5 = st.columns(2)

with col4:
    response_time = st.number_input(
        "Response Time (seconds)",
        min_value=0.0
    )

with col5:
    remarks = st.text_area("Customer Remarks")


st.divider()


# -----------------------------
# MODEL SELECTION + BUTTON
# -----------------------------

btn_col1, btn_col2 = st.columns([1,1])

with btn_col1:

    model_option = st.selectbox(
        "Select Model",
        ["LSTM", "BiLSTM"]
    )

with btn_col2:

    predict_button = st.button("🚀 Predict CSAT Score")


model = load_model_selected(model_option)


# -----------------------------
# ENCODING MAPS
# -----------------------------

channel_map = {"Chat":0,"Email":1,"Phone":2,"Social Media":3}

category_map = {
    "Delivery":0,
    "Product Issue":1,
    "Return":2,
    "Payment":3
}

sub_map = {
    "Late Delivery":0,
    "Damaged Item":1,
    "Wrong Item":2,
    "Refund Issue":3
}

city_map = {
    "Delhi":0,
    "Mumbai":1,
    "Bangalore":2,
    "Chennai":3
}

product_map = {
    "Electronics":0,
    "Fashion":1,
    "Home":2,
    "Beauty":3
}

tenure_map = {
    "0-6 months":0,
    "6-12 months":1,
    "1-2 years":2
}

shift_map = {
    "Morning":0,
    "Evening":1,
    "Night":2
}


# -----------------------------
# PREDICTION
# -----------------------------

if predict_button:

    if remarks.strip() == "":
        st.warning("Please enter customer remarks")

    else:

        seq = tokenizer.texts_to_sequences([remarks])

        padded = pad_sequences(seq, maxlen=max_len)


        numeric_data = np.array([
            [item_price, handling_time, response_time]
        ])

        scaled_numeric = scaler.transform(numeric_data)


        struct_data = np.array([[
            channel_map[channel],
            category_map[category],
            sub_map[sub_category],
            city_map[city],
            product_map[product_category],
            tenure_map[tenure],
            shift_map[shift],
            scaled_numeric[0][0],
            scaled_numeric[0][1],
            scaled_numeric[0][2]
        ]])


        prediction = model.predict([padded, struct_data])

        predicted_class = np.argmax(prediction)

        csat_score = predicted_class + 1

        confidence = prediction[0][predicted_class] * 100


        res1, res2 = st.columns(2)

        with res1:

            st.success(f"⭐ Predicted CSAT Score: {csat_score}")

            st.metric(
                "Confidence Score",
                f"{confidence:.2f}%"
            )

            st.write(f"Model Used: **{model_option}**")

        with res2:

            st.subheader("Prediction Probability")

            prob_df = pd.DataFrame({
                "CSAT":[1,2,3,4,5],
                "Probability":prediction[0]
            })

            st.bar_chart(prob_df.set_index("CSAT"))