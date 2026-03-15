# 📊 Customer Satisfaction (CSAT) Prediction Dashboard

This project builds a **Deep Learning–based Customer Satisfaction Prediction System** that predicts the **CSAT Score (1–5)** using customer service interaction data.

The system uses **Natural Language Processing (NLP)** and **Deep Learning models (LSTM & BiLSTM)** to analyze **customer remarks and structured service data**.

A **Streamlit dashboard** allows users to input interaction details and receive real-time CSAT predictions.

---

## 🚀 Live Demo (Streamlit App)

🔗 **Streamlit Demo:**
👉 https://deep-csat-prediction.streamlit.app/

# 🚀 Project Overview

Customer Satisfaction (CSAT) is a key metric used by organizations to measure **customer experience and service quality**.

This project aims to:

* Predict customer satisfaction using **historical interaction data**
* Analyze **customer remarks using NLP**
* Combine **text data + structured features**
* Compare **LSTM and BiLSTM models**
* Provide a **real-time prediction dashboard**

---

# 🧠 Machine Learning Approach

The system combines two types of data:

### 1️⃣ Text Data

Customer remarks are processed using:

* Tokenization
* Sequence padding
* Word embeddings
* LSTM / BiLSTM networks

### 2️⃣ Structured Data

Additional interaction features are used such as:

* Channel name
* Category
* Sub-category
* Customer city
* Product category
* Agent tenure
* Agent shift
* Item price
* Handling time
* Response time

These features are **scaled using StandardScaler** before training.

---

# 🏗️ Model Architecture

## LSTM Model

Text pipeline:

Customer Remarks → Tokenizer → Embedding → LSTM → Dense

Structured pipeline:

Structured Features → Dense Layer

Final prediction:

Concatenation → Dense → Softmax (CSAT Score)

---

## BiLSTM Model

BiLSTM improves performance by reading sequences **forward and backward**.

Customer Remarks → Embedding → **Bidirectional LSTM** → Dense

Structured features are merged with text embeddings to generate the final prediction.

---

# 📊 Model Output

The model predicts:

* **CSAT Score (1–5)**
* **Confidence Score**
* **Probability distribution for each CSAT class**

Example:

Predicted CSAT Score: 4
Confidence Score: 82.45%

---

# 🖥️ Streamlit Dashboard

The project includes a **Streamlit dashboard** that allows users to:

* Enter customer interaction details
* Select the model (LSTM / BiLSTM)
* Predict CSAT score
* View confidence score
* Visualize probability distribution

Dashboard features:

* Grid-based input layout
* Dynamic Category → Sub-category mapping
* Model selection
* Real-time predictions

---

# 📂 Project Structure

```
deep_csat_prediction/

│
├── app.py
├── lstm_model.h5
├── bilstm_model.h5
├── tokenizer.pkl
├── scaler.pkl
├── dataset.csv
├── requirements.txt
└── README.md
```

---

# ⚙️ Installation

### 1️⃣ Clone Repository

```
git clone https://github.com/yourusername/csat-prediction-dashboard.git
```

```
cd csat-prediction-dashboard
```

---

### 2️⃣ Install Dependencies

```
pip install -r requirements.txt
```

Required libraries:

* streamlit
* tensorflow
* numpy
* pandas
* scikit-learn

---

### 3️⃣ Run Streamlit App

```
streamlit run app.py
```

The dashboard will open in your browser.

---

# 📈 Example Prediction Workflow

1. User enters interaction details
2. Customer remarks are tokenized
3. Structured features are scaled
4. Selected model (LSTM / BiLSTM) processes input
5. CSAT score is predicted
6. Confidence score and probability chart are displayed

---

# 🔍 Features Implemented

✔ LSTM Deep Learning model
✔ BiLSTM model for improved text understanding
✔ Text preprocessing pipeline
✔ Feature scaling
✔ Dynamic category–subcategory validation
✔ Streamlit prediction dashboard
✔ Confidence score display
✔ Model selection interface
✔ Probability distribution visualization

---

# 📊 Future Improvements

Potential improvements include:

* Transformer models (BERT)
* Sentiment analysis integration
* Explainable AI (SHAP)
* Batch CSV predictions
* Model performance dashboard
* API deployment with FastAPI

---

# 👨‍💻 Author

Developed as a **Machine Learning / NLP project** for customer experience analytics.

---

# 📜 License

This project is open-source and available for educational and research purposes.
