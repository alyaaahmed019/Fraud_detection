# Fraud Detection System

### IEEE-CIS Transaction Fraud Detection (End-to-End ML Project)

---

## 📌 Overview

This project presents a full **end-to-end machine learning pipeline** for detecting fraudulent financial transactions using the IEEE-CIS dataset.

It covers:

* 📊 Deep data analysis (EDA)
* 🧹 Data cleaning & preprocessing
* ⚙️ Feature engineering
* 🌲 Model training using Random Forest
* 🖥️ Deployment via an interactive Streamlit dashboard

The goal is to build a system that can **identify suspicious transactions in real-time based on behavioral patterns and historical risk signals**.

---

## 📊 Dataset

* Source: IEEE-CIS Fraud Detection Dataset
* Size: ~590,000 transactions
* Includes:

  * Transaction data
  * Identity data
  * High-dimensional features (V, C, D, M, ID)

---

## 🔍 Exploratory Data Analysis (EDA)

Key analysis steps:

* Distribution of transaction amounts
* Time-based behavior (hour/day patterns)
* Fraud vs. non-fraud comparison
* Missing value analysis (many features >80% null)
* Identification of high-risk segments:

  * Cards
  * Email domains
  * Products

📌 Insight: Fraud is highly **imbalanced** and often linked to **behavioral anomalies**, not just single features.

---

## 🧹 Data Cleaning

* Dropped columns with **>80% missing values**
* Handled missing values:

  * Median (numerical)
  * Mode (categorical)
* Standardized feature names and formats
* Removed redundant/noisy features

---

## ⚙️ Feature Engineering

Created powerful features to capture behavior:

### 🕒 Time Features

* `hour_of_day`
* `day_of_week`
* `is_night`

### 💰 Transaction Features

* `log_amt`
* `amt_vs_card_avg_ratio`

### 💳 Card Features

* `card1_txn_count`
* `card1_avg_amt`
* `card1_historical_fraud_rate`

### 📧 Email Features

* `email_txn_count`
* `email_historical_fraud_rate`

### 🔢 Encoded Features

* Binary encoding (M features)
* One-hot encoding for categorical variables

---

## ⚙️ Preprocessing Pipeline

Built using `ColumnTransformer`:

* **Numerical pipeline**

  * Median imputation
  * Standard scaling

* **Categorical pipeline**

  * Mode imputation
  * One-hot encoding

✔ Ensures consistency between training and inference
✔ Prevents data leakage

---

## 🌲 Model: Random Forest

* Algorithm: **Random Forest Classifier**
* Reason for choice:

  * Handles high-dimensional data well
  * Robust to noise and missing values
  * Captures non-linear relationships

### 📈 Training Strategy

* Stratified train-test split (preserves fraud ratio)
* Trained on processed features
* Evaluated using:

  * Accuracy
  * Precision / Recall
  * ROC-AUC

---

## ⚠️ Challenge: Class Imbalance

Fraud cases are extremely rare:

* ~99% normal
* ~1% fraud

Handled by:

* Careful evaluation metrics
* Probability-based prediction (not just class labels)
* Threshold tuning (recommended)

---

## 🖥️ Deployment (Streamlit)

An interactive dashboard allows users to:

* Input transaction details
* Simulate real-world scenarios
* Get:

  * Fraud prediction
  * Fraud probability score

### 💡 UX Features

* Simplified input fields (~10 key features)
* Auto-filled defaults for missing data
* Real-time prediction

---

## 🔮 How It Works

```text
User Input
   ↓
Default Value Filling
   ↓
Preprocessing (ColumnTransformer)
   ↓
Random Forest Model
   ↓
Fraud Probability + Prediction
```

---

## 📂 Project Structure

```
├── data/
├── notebooks/
├── app.py                # Streamlit app
├── preprocessor.pkl
├── model_rf.pkl
├── defaults.pkl
├── features.pkl
└── README.md
```

---

## 🚀 Results

The system successfully:

* Detects fraud based on behavioral anomalies
* Handles large, noisy datasets
* Provides real-time predictions

---

## 🧠 Key Insights

* Fraud detection depends on **patterns, not single features**
* Historical behavior is a strong signal
* Feature engineering significantly improves performance

---

## 🔧 Future Improvements

* 🔍 Feature importance visualization
* 📊 SHAP explainability
* ⚖️ Better class balancing (SMOTE, weighting)
* 🎯 Threshold optimization
* ☁️ Cloud deployment (Streamlit Cloud / AWS)

---

## 👨‍💻 Author

Developed as part of a machine learning project focused on **real-world fraud detection systems**.

