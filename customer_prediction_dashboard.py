import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Churn Prediction", layout="centered")
st.title("🔮 Telco Customer Churn Predictor")

st.write("Fill out the customer details below to predict churn probability.")

with st.form("predict_form"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", ["Yes", "No"])
    partner = st.selectbox("Has Partner", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0)
    total_charges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=1500.0)
    submitted = st.form_submit_button("Predict")

if submitted:
    input_data = pd.DataFrame({
        'gender': [gender],
        'SeniorCitizen': [1 if senior == 'Yes' else 0],
        'Partner': [partner],
        'Dependents': [dependents],
        'tenure': [tenure],
        'PhoneService': [phone_service],
        'MultipleLines': [multiple_lines],
        'InternetService': [internet_service],
        'OnlineSecurity': [online_security],
        'Contract': [contract],
        'PaymentMethod': [payment_method],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges]
    })

    try:
        model = joblib.load("churn_model.pkl")
        model_columns = joblib.load("model_columns.pkl")
        input_df = pd.get_dummies(input_data)
        input_df = input_df.reindex(columns=model_columns, fill_value=0)
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        st.subheader("Prediction Result")
        st.write("Churn Prediction:", "Yes" if prediction == 1 else "No")
        st.write("Probability of Churn: {:.2f}%".format(probability * 100))

    except Exception as e:
        st.error("Model or preprocessing pipeline not found. Please upload 'churn_model.pkl' and 'preprocess_pipeline.pkl'.")
        print(e)
