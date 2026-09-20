# Banking Fraud Detection — Streamlit Batch Scoring App

A portfolio project for **credit-card fraud screening** using a saved Random Forest model and a Streamlit batch-scoring interface.

The app accepts transaction CSV files in the classic Credit Card Fraud Detection feature format (**Time, V1–V28, Amount**), validates the schema, applies the saved scaler/model, and returns transactions flagged for review.

> **Important:** this is an educational ML prototype. A prediction is a screening signal, not proof of fraud and not a production payment-blocking system.

## Why this project exists

Fraud detection is a highly imbalanced classification problem. A useful system must consider more than accuracy: recall matters for catching fraud, while precision matters because excessive false positives create unnecessary investigations and customer friction.

This repository demonstrates:

- imbalanced fraud modelling
- feature scaling
- SMOTE-based training workflow in the original modelling work
- Logistic Regression vs Random Forest comparison
- saved model artifacts
- batch inference from CSV
- input validation and downloadable scored results
- a deployable Streamlit interface

## Dataset

The modelling work uses the Credit Card Fraud Detection dataset:

- 284,807 transactions
- 492 fraud cases in the original dataset
- fraud rate of roughly 0.17%
- anonymized PCA features V1–V28 plus Time and Amount

The raw source dataset is not committed to this repository.

## Recorded model results

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.058 | 0.918 | 0.109 | 0.946 |
| Random Forest | 0.406 | 0.837 | 0.547 | 0.917 |

The saved Streamlit application uses the **Random Forest** artifact because it provided a substantially better precision/recall balance in the original experiment.

These results are experiment-specific and do not establish production performance.

## Run locally

Use a clean environment rather than the original machine-wide Anaconda export.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Then upload a CSV with these columns:

```text
Time, V1, V2, ... V28, Amount
```

A `Class` column may be present; it is ignored during inference.

## What the app does

1. validates required columns
2. rejects missing/non-numeric model inputs
3. reorders inputs to the training schema
4. applies the saved scaler
5. scores rows with the saved Random Forest
6. summarizes flagged transactions
7. lets the user download the scored CSV

## Deployment

This repository is configured for a Render web service.

**Build**

```bash
pip install -r requirements.txt
```

**Start**

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port $PORT
```

A live URL should be added only after the hosted service has been verified.

## API integration

**No external API is integrated.**

The app performs local inference using model files in `models/`. It does not connect to a bank core system, card network, payment gateway, or transaction API.

If this were extended into a service consumed by other applications, the appropriate next step would be a small **FastAPI inference service** with a documented request/response schema—not adding an API simply for appearance.

## Repository structure

```text
banking-fraud-detection/
├── app.py
├── models/
│   ├── random_forest_model.pkl
│   └── scaler.pkl
├── notebooks/
├── PROJECT_STATUS.md
├── render.yaml
├── requirements.txt
└── README.md
```

## Limitations

- public benchmark data, not live institutional transaction data
- model artifacts depend on the preprocessing/training environment
- no probability calibration or institution-specific review threshold
- no drift monitoring, explainability workflow, authentication, or audit database
- no real-time transaction integration

## Portfolio role

This is an **older fraud project upgraded into an interactive batch-scoring demo**. The newer `financial-fraud-detection-dashboard` repository is the stronger end-to-end fraud engineering project and should remain the primary flagship.
