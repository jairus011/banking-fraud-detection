import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("models/random_forest_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# App title
st.title("Banking Fraud Detection System")

st.write(
    "Upload a transaction CSV file to detect fraudulent transactions."
)

# Upload CSV
uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    # Read uploaded file
    data = pd.read_csv(uploaded_file)

    # Remove target column if present
    if "Class" in data.columns:
        data = data.drop("Class", axis=1)

    st.subheader("Uploaded Data")
    st.write(data.head())

    # Scale data
    scaled_data = scaler.transform(data)

    # Predict
    predictions = model.predict(scaled_data)

    # Add predictions
    data["Prediction"] = predictions

    # Convert predictions to labels
    data["Prediction"] = data["Prediction"].map({
        0: "Normal",
        1: "Fraud"
    })

    st.subheader("Prediction Results")
    st.write(data)

    # Fraud count
    fraud_count = (data["Prediction"] == "Fraud").sum()

    st.success(
        f"Detected {fraud_count} fraudulent transactions."
    )