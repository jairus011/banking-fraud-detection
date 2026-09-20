from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Banking Fraud Detection",
    page_icon="🛡️",
    layout="wide",
)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "random_forest_model.pkl"
SCALER_PATH = ROOT / "models" / "scaler.pkl"

EXPECTED_FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        raise FileNotFoundError(
            "Model artifacts are missing. Expected models/random_forest_model.pkl "
            "and models/scaler.pkl."
        )
    return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)


st.title("🛡️ Banking Fraud Detection")
st.caption("Batch transaction screening with a saved Random Forest model")

st.warning(
    "Educational portfolio prototype only. A model prediction is a screening flag, "
    "not proof that a transaction is fraudulent."
)

with st.expander("Expected CSV format"):
    st.write(
        "Upload the Credit Card Fraud Detection feature schema: Time, V1–V28, and Amount. "
        "A Class column is allowed and will be ignored during prediction."
    )
    template = pd.DataFrame(columns=EXPECTED_FEATURES)
    st.download_button(
        "Download empty CSV template",
        data=template.to_csv(index=False).encode("utf-8"),
        file_name="fraud_scoring_template.csv",
        mime="text/csv",
    )

try:
    model, scaler = load_artifacts()
except Exception as exc:
    st.error(f"Could not load the saved model: {exc}")
    st.stop()

uploaded_file = st.file_uploader("Upload transaction CSV", type=["csv"])

if uploaded_file is None:
    st.info("Upload a CSV to run batch scoring.")
    st.stop()

try:
    raw = pd.read_csv(uploaded_file)
except Exception as exc:
    st.error(f"Could not read the CSV: {exc}")
    st.stop()

if raw.empty:
    st.error("The uploaded CSV contains no rows.")
    st.stop()

input_data = raw.drop(columns=["Class"], errors="ignore").copy()
missing = [c for c in EXPECTED_FEATURES if c not in input_data.columns]
extra = [c for c in input_data.columns if c not in EXPECTED_FEATURES]

if missing:
    st.error("Missing required columns: " + ", ".join(missing))
    st.stop()

if extra:
    st.info(
        "Ignoring extra columns that are not part of the trained feature schema: "
        + ", ".join(extra)
    )

input_data = input_data[EXPECTED_FEATURES]

non_numeric = [
    c for c in EXPECTED_FEATURES
    if not pd.api.types.is_numeric_dtype(input_data[c])
]
if non_numeric:
    st.error(
        "These required columns must be numeric: " + ", ".join(non_numeric)
    )
    st.stop()

if input_data.isna().any().any():
    bad = input_data.columns[input_data.isna().any()].tolist()
    st.error(
        "Missing values were found in: " + ", ".join(bad)
        + ". Clean or impute them before scoring."
    )
    st.stop()

try:
    scaled = scaler.transform(input_data)
    predictions = model.predict(scaled)
except Exception as exc:
    st.error(
        "Prediction failed. The uploaded schema may not match the saved model, "
        f"or the serialized model may be incompatible with this runtime. Details: {exc}"
    )
    st.stop()

scored = raw.copy()
scored["Prediction"] = pd.Series(predictions).map({0: "Normal", 1: "Fraud"})
scored["Fraud_Flag"] = predictions.astype(int)

fraud_count = int((predictions == 1).sum())
total = len(predictions)
fraud_rate = fraud_count / total if total else 0.0

m1, m2, m3 = st.columns(3)
m1.metric("Transactions scored", f"{total:,}")
m2.metric("Flagged for review", f"{fraud_count:,}")
m3.metric("Flag rate", f"{fraud_rate:.2%}")

st.subheader("Scored transactions")
st.dataframe(scored, use_container_width=True, hide_index=True)

st.download_button(
    "Download scored CSV",
    data=scored.to_csv(index=False).encode("utf-8"),
    file_name="fraud_scored_transactions.csv",
    mime="text/csv",
)

st.caption(
    "This application performs local model inference only. It does not call a bank, "
    "payment network, or external API, and it does not block transactions."
)
