# Banking Fraud Detection System

## Project Overview

This project develops a machine learning-based fraud detection system for banking transactions using the Credit Card Fraud Detection dataset.

The goal of the project is to identify fraudulent transactions while minimizing false positives and improving financial security.

Fraud detection is a critical problem in the banking and financial sector because fraudulent activities can result in major financial losses and reduced customer trust.

---

# Business Problem

Banks process millions of transactions daily. Detecting fraudulent transactions manually is inefficient and costly.

This project applies machine learning techniques to:
- automatically identify suspicious transactions,
- reduce financial losses,
- improve fraud investigation efficiency,
- support intelligent banking security systems.

---

# Dataset Information

Dataset used:
- Credit Card Fraud Detection Dataset

Dataset characteristics:
- 284,807 transactions
- Highly imbalanced dataset
- Fraudulent transactions represent approximately 0.17% of all transactions

Features:
- Numerical transaction features (V1–V28)
- Transaction Amount
- Transaction Time
- Target variable:
  - 0 = Normal transaction
  - 1 = Fraudulent transaction

---

# Project Workflow

## 1. Exploratory Data Analysis (EDA)
- Fraud distribution analysis
- Transaction amount analysis
- Class imbalance visualization
- Missing value checks

## 2. Data Preprocessing
- Train-test split
- Feature scaling using StandardScaler
- Class balancing using SMOTE

## 3. Machine Learning Modeling
Two supervised learning models were trained:

### Logistic Regression
### Random Forest Classifier

---

# Model Evaluation Metrics

The models were evaluated using:
- Precision
- Recall
- F1-score
- ROC-AUC

---

# Final Results

| Model | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.058 | 0.918 | 0.109 | 0.946 |
| Random Forest | 0.406 | 0.837 | 0.547 | 0.917 |

---

# Best Performing Model

## Random Forest Classifier

Random Forest achieved the best overall balance between:
- fraud detection capability,
- false positive reduction,
- practical usability in banking environments.

Although Logistic Regression achieved higher recall, it produced excessive false positives.

---

# Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Jupyter Notebook
- Git & GitHub

---

# Project Structure

```text
banking-fraud-detection/
│
├── data/
├── models/
│   ├── random_forest_model.pkl
│   └── scaler.pkl
│
├── notebooks/
│   ├── fraud_detection_eda.ipynb
│   ├── fraud_detection_preprocessing.ipynb
│   └── fraud_detection_modeling.ipynb
│
├── reports/
├── src/
└── README.md