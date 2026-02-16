# ==========================
# Streamlit Frontend: Model Stealing Lab
# ==========================

import streamlit as st
import numpy as np
import os
import tempfile
import secrets
from tensorflow.keras.models import load_model
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# --------------------------
# Page Config
# --------------------------
st.set_page_config(
    page_title="Model Stealing Lab",
    layout="centered"
)

# --------------------------
# Session-Based Auth Token
# --------------------------
def get_or_create_auth_token():
    if "auth_token" not in st.session_state:
        st.session_state.auth_token = secrets.token_hex(16)  # 32-character token
    return st.session_state.auth_token

auth_token = get_or_create_auth_token()

# --------------------------
# Load Dataset for Evaluation
# --------------------------
@st.cache_data
def load_test_data():
    data = load_diabetes()
    X, y = data.data, (data.target > 140).astype(int)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    _, X_test, _, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    return X_test, y_test

X_test, y_test = load_test_data()

# --------------------------
# Load Original Protected Model
# --------------------------
@st.cache_resource
def load_original_model():
    return load_model("backend_model.h5")  # Must be placed in the same folder

# --------------------------
# Load Uploaded Stolen Model
# --------------------------
def load_uploaded_model(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
        tmp.write(uploaded_file.read())
        model_path = tmp.name
    model = load_model(model_path)
    os.remove(model_path)
    return model

# --------------------------
# Evaluate a Keras Model
# --------------------------
def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    preds_binary = (preds > 0.5).astype(int).flatten()
    return accuracy_score(y_test, preds_binary)

# --------------------------
# Streamlit UI
# --------------------------

st.title("🧠 Model Stealing Lab")

st.markdown("""
Welcome to the **Model Stealing Lab**. This lab demonstrates how to extract knowledge from a black-box machine learning model using adaptive querying techniques.

---

### 🔐 Authentication Token

Use the token below when querying the black-box model API from your notebook:
""")

st.code(f"Authorization: Bearer {auth_token}", language="bash")

st.markdown("""
This token is required in the `Authorization` header of your API requests.

---

### 📡 Model API Endpoint

Query the model hosted in the lab backend:

POST /predict

            """)


# 📓 Notebook Link
st.markdown("### 🧪 Part 1: Perform Model Stealing")

st.link_button("📓 Open Adaptive Querying Notebook", "https://colab.research.google.com/drive/182PZ7qdyq3Yest7fqcJ3DqqYkjMTkyxm")


# --------------------------
# 📤 Part 2: Upload & Evaluate Stolen Model
# --------------------------

st.markdown("---")
st.subheader("📤 Part 2: Upload Your Stolen Model")

uploaded_file = st.file_uploader("Upload your `stolen_model.h5`", type=["h5"])
evaluate_button = st.button("✅ Upload and Compare to Original")

if evaluate_button and uploaded_file:
    try:
        # Evaluate stolen model
        stolen_model = load_uploaded_model(uploaded_file)
        stolen_acc = evaluate_model(stolen_model, X_test, y_test)
        st.success("✅ Stolen model evaluated successfully.")
        st.metric("Stolen Model Accuracy", f"{stolen_acc:.4f}")

        # Evaluate original model
        try:
            original_model = load_original_model()
            original_acc = evaluate_model(original_model, X_test, y_test)
            st.metric("Original Model Accuracy", f"{original_acc:.4f}")
        except Exception as e:
            st.warning("⚠️ Could not load original model for comparison.")
            st.error(str(e))

    except Exception as e:
        st.error(f"❌ Error loading or evaluating stolen model: {e}")

elif evaluate_button and not uploaded_file:
    st.warning("📂 Please upload a model file before clicking 'Upload and Compare'.")
