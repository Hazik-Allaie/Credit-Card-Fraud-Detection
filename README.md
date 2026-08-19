# Credit Card Fraud Detection 💳🔍

> **End-to-end machine learning project** — from raw imbalanced data to a deployed Streamlit web app that flags fraudulent transactions in real time.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red.svg)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange.svg)](https://xgboost.readthedocs.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5+-gold.svg)](https://scikit-learn.org/)
[![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-0.12+-green.svg)](https://imbalanced-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📖 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Model Performance](#-model-performance)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results Summary](#-results-summary)
- [Deployment](#-deployment)
- [Future Work](#-future-work)
- [Author](#-author)

---

## 🎯 Project Overview

Credit card fraud is a multi-billion dollar problem. The challenge: **fraud is rare** (≈0.17% of transactions), so standard accuracy metrics are meaningless — a model that predicts "legit" for everything achieves 99.83% accuracy but catches zero fraud.

This project tackles the full ML lifecycle:

1. **Exploratory Data Analysis** — understand the highly imbalanced, PCA-transformed features
2. **Imbalance Handling** — SMOTE oversampling to give the model enough fraud examples to learn from
3. **Model Training & Comparison** — Logistic Regression → Random Forest → XGBoost (winner)
4. **Hyperparameter Tuning & Threshold Analysis** — RandomizedSearchCV + decision-threshold sweep
5. **Precision-Recall Evaluation** — PR curves & Average Precision (AP = 0.877) over misleading ROC-AUC
5. **Production Pipeline** — `StandardScaler → SMOTE → XGBoost` packaged as a single `joblib` artifact
6. **Web App** — Streamlit interface with **manual entry** and **CSV batch upload**
7. **Deployment** — Streamlit Cloud + GitHub

**Key result:** A model that catches **85% of fraud** with only **31 false alarms per 56,864 legit transactions**, deployed as a live web app.

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| **Source** | [Machine Learning Group – ULB](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (`creditcard.csv`) |
| **Size** | 284,807 transactions × 31 columns |
| **Features** | `Time`, `Amount`, `V1`–`V28` (PCA-transformed, anonymized) |
| **Target** | `Class` (0 = Legitimate, 1 = Fraud) |
| **Class Balance** | 284,315 legit (99.83%) vs 492 fraud (0.17%) |
| **Missing Values** | None |

> **Note:** The original dataset is not included in this repo due to size (≈144 MB). Download it from Kaggle and place `creditcard.csv` in the project root before running the notebook.

---

## 🧪 Methodology

### Phase 1: Exploratory Data Analysis
- Verified class imbalance (99.83% / 0.17%)
- Confirmed no missing values
- Visualized `Amount` distribution (heavy-tailed, requires scaling)
- Noted `V1`–`V28` are already PCA components — no further feature engineering needed

### Phase 2: Baseline Models with Imbalance Handling
| Model | Fraud Recall | False Alarms | Precision | F1 |
|-------|-------------|--------------|-----------|-----|
| Logistic Regression (baseline) | 64% | 13 | 0.83 | 0.72 |
| Logistic + SMOTE | 92% | 1,458 | 0.06 | 0.11 |
| Random Forest + SMOTE | 82% | 17 | 0.82 | 0.82 |
| **XGBoost + SMOTE** | **89%** | **32** | **0.73** | **0.80** |

**Winner:** XGBoost + SMOTE — best recall/false-alarm tradeoff for production.

### Phase 3: Tuning & Threshold Evaluation
- **RandomizedSearchCV** (3-fold CV on SMOTE data): best params `n_estimators=200, max_depth=7, learning_rate=0.1, subsample=0.9`
- **Honest finding:** Tuned model **overfit to SMOTE** — on real imbalanced test data it caught **1 fewer fraud** (88% vs 89%) with same false alarms. **Default XGBoost retained.**
- **Threshold sweep** (0.5 → 0.1): Lowering threshold barely gained frauds (+1 max) but false alarms exploded (32 → 109). **Default 0.5 kept.**
- **Precision-Recall Curve:** AP = **0.8774** — curve flat near 1.0 precision up to ~80% recall. Strong signature on imbalanced data.

### Phase 4: Production Pipeline
```python
Pipeline([
    ("scaler", StandardScaler()),           # fits on RAW Amount & Time
    ("smote", SMOTE(random_state=42)),      # runs ONLY during .fit()
    ("model", XGBClassifier(...))           # default params, best honest result
])
```
- **Critical:** Uses `imblearn.pipeline.Pipeline` (not sklearn's) — SMOTE needs `.fit_resample()`
- **Self-contained:** Single `model.pkl` expects **raw** features in order `V1–V28, Amount, Time`
- **Verification:** Load-back test confirms identical predictions

### Phase 5: Streamlit Web App
- **Manual Entry:** 30-input form → instant verdict + probability
- **CSV Upload:** Batch prediction → adds `Prediction` & `Fraud Probability` columns
- **Tested on real data:** 3 fraud + 3 legit rows → 2/3 frauds caught at 99%+ confidence, 1 clean miss (0.1% prob), 0 false alarms

---

## 📈 Model Performance

| Metric | Value | Notes |
|--------|-------|-------|
| **Fraud Recall (Test)** | 85% | 83 / 98 frauds caught |
| **False Alarms** | 31 | per 56,864 legit transactions |
| **Precision (Fraud)** | 0.73 | |
| **F1 (Fraud)** | 0.80 | |
| **Average Precision (PR-AUC)** | 0.877 | Phase 3; 0.868 on retrained pipeline |
| **Decision Threshold** | 0.5 | Default — optimal per threshold sweep |

> **Why not ROC-AUC?** On 99.83% imbalance, ROC-AUC is misleading (FPR denominator dominated by legit class). **Precision-Recall is the honest metric.**

---

## 📁 Project Structure

```
credit-card-fraud-detection/
├── app.py                  # Streamlit web app (entry point)
├── model.pkl               # Trained Pipeline (scaler + SMOTE + XGBoost)
├── fraud_detection.ipynb   # Full notebook: EDA → modeling → evaluation
├── test.csv                # 6-row test file (3 fraud, 3 legit) for demo
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── creditcard.csv          # (NOT in repo — download from Kaggle)
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.10+
- Git (for cloning)

### Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/credit-card-fraud-detection.git
cd credit-card-fraud-detection

# 2. Create & activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the dataset
#    → Go to https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
#    → Download creditcard.csv
#    → Place it in this project root folder
```

### requirements.txt
```text
streamlit>=1.35
joblib>=1.4
scikit-learn>=1.5
imbalanced-learn>=0.12
xgboost>=2.0
pandas>=2.2
numpy>=1.26
```

---

## 🚀 Usage

### Run the Web App
```bash
streamlit run app.py
```
- Opens `http://localhost:8501` in your browser
- Choose **Manual Entry** (30 boxes) or **Upload CSV**
- CSV must contain columns: `V1`–`V28`, `Amount`, `Time` (raw values)

### Run the Notebook
```bash
jupyter notebook fraud_detection.ipynb
```
- Execute cells top-to-bottom
- Reproduces all phases: EDA, modeling, tuning, pipeline, verification

### Quick Test (no dataset needed)
```bash
streamlit run app.py
# → Select "Upload CSV" → upload the included test.csv
# → Shows 2 frauds caught at 99%+, 1 clean miss, 0 false alarms
```

---

## 📋 Results Summary

| Phase | Deliverable | Status |
|-------|-------------|--------|
| 1 | EDA + class balance visualization | ✅ |
| 2 | XGBoost + SMOTE (89% recall, 32 FA) | ✅ |
| 3 | Threshold sweep + PR curve (AP=0.877) | ✅ |
| 4 | `model.pkl` (self-contained Pipeline) | ✅ |
| 5 | `app.py` (manual + CSV upload) | ✅ |
| 6 | Streamlit Cloud deploy + GitHub | 🔄 |

**Honest takeaway:** The model doesn't catch every fraud — no model does. But when it flags one, it's 99%+ sure. That's the signal a fraud team can act on.

---

## ☁️ Deployment

### Streamlit Cloud (Free, 2-minute deploy)

1. Push this repo to GitHub (see `Installation` → step 1)
2. Go to [share.streamlit.io](https://share.streamlit.io) → Sign in with GitHub
3. Click **New app** → Select your repo → Branch: `main` → Main file: `app.py`
4. Click **Deploy!**
5. Get a live URL: `https://credit-card-fraud-detection-byhaziq.streamlit.app/

> **Note:** `model.pkl` (≈50 MB) is tracked via Git LFS or included directly. Streamlit Cloud installs `requirements.txt` automatically.

### Local Docker (optional)
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

## 🔮 Future Work

- [ ] **Cost-sensitive learning** — weigh false negatives (missed fraud) vs false positives (blocked legit)
- [ ] **Explainability** — SHAP values for individual predictions in the app
- [ ] **Drift monitoring** — track feature distributions over time
- [ ] **Ensemble** — combine XGBoost + LightGBM + Neural Net for robustness
- [ ] **Real-time scoring** — Kafka / Redis pipeline for sub-second latency
- [ ] **A/B testing framework** — compare model versions in production

---


## 👤 Author

**Hazik Allaie**  
🔗 [GitHub](https://github.com/Hazik-Allaie) · [LinkedIn](https://linkedin.com/in/hazik-allaie-aa81a3333)  
📧 allaiehaziq786@gmail.com

> Built as a hands-on ML portfolio project — every phase documented, every decision explained, every result honest.
**Live url** : https://credit-card-fraud-detection-byhaziq.streamlit.app/ 
---

⭐ **If this project helped you learn, consider starring the repo!**  
🐛 **Found an issue?** Open a GitHub issue — I'd love to hear from you.