# ===== CREDIT CARD FRAUD DETECTOR — app.py =====
 
import joblib
import pandas as pd
import streamlit as st
 
# --- Load the trained model once at startup ---
pipeline = joblib.load('model.pkl')
 
# --- Page setup ---
st.set_page_config(page_title="Credit Card Fraud Detector", layout="centered")
st.title("💳 Credit Card Fraud Detector")
st.write("Upload a CSV or enter a transaction manually — the model predicts the chance of fraud.")
 
# The 30 features the model was trained on, in this exact order
features = [f'V{i}' for i in range(1, 29)] + ['Amount', 'Time']
 
# --- Choose how to check ---
mode = st.radio("How do you want to check a transaction?", ["Manual Entry", "Upload CSV"])
 
if mode == "Manual Entry":
    st.subheader("Enter transaction details")
 
    with st.form("manual_form"):
        cols = st.columns(3)              # drop the 30 boxes into 3 columns
        inputs = {}
        for i, feat in enumerate(features):
            inputs[feat] = cols[i % 3].number_input(feat, value=0.0, format="%.6f")
 
        submitted = st.form_submit_button("Check Transaction")
 
    if submitted:
        sample = pd.DataFrame([inputs])   # one row, all 30 inputs
        sample = sample[features]         # force model's exact column order
 
        proba = pipeline.predict_proba(sample)[0][1]   # P(fraud)
        pred = pipeline.predict(sample)[0]
 
        if pred == 1:
            st.error(f"🚨 FRAUD — {proba * 100:.2f}% probability of fraud")
        else:
            st.success(f"✅ LEGIT — {(1 - proba) * 100:.2f}% legit")
 
else:
    st.subheader("Upload a CSV file")
    st.info("CSV columns must match the training data: V1–V28, Amount, Time "
            "(raw values — the model scales them itself). One row per transaction.")
 
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
 
    if uploaded is not None:
        data = pd.read_csv(uploaded)
 
        missing = [f for f in features if f not in data.columns]
        if missing:
            st.error(f"Missing columns from CSV: {missing}")
        else:
            data = data[features]                       # reorder, drop extras
            preds = pipeline.predict(data)
            probas = pipeline.predict_proba(data)[:, 1]
 
            result = data.copy()
            result["Prediction"] = ["FRAUD" if p == 1 else "LEGIT" for p in preds]
            result["Fraud Probability"] = probas
 
            st.subheader("Predictions")
            st.dataframe(result)
            st.success(f"Checked {len(data)} transactions — {sum(preds)} flagged as fraud.")