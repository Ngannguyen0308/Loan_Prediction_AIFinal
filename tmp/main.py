import streamlit as st
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open("./model/loan_status_predictor.pkl", "rb"))

st.title("Loan Status Predictor")

# Add UI components (input fields, buttons, etc.)
credit_history = st.slider("Credit History", 0.0, 1.0, 0.5)
loan_amount = st.slider("Loan Amount", 0, 1000, 500)
applicant_income = st.slider("Applicant Income", 0, 100000, 50000)
coapplicant_income = st.slider("Coapplicant Income", 0, 100000, 50000)
dependents = st.selectbox("Dependents", [0, 1, 2, 3])

# Perform prediction using the loaded model
input_data = np.array([[credit_history, loan_amount, applicant_income, coapplicant_income, dependents]])
prediction = model.predict(input_data)

# Display the prediction
st.subheader("Prediction")
st.write("Loan Status:", "Approved" if prediction[0] == 'Y' else "Not Approved")
