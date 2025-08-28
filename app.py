import streamlit as st
import pandas as pd
import joblib

# Load the trained model pipeline
try:
    pipeline = joblib.load('fraud_model.pkl')
except FileNotFoundError:
    st.error("Error: The 'fraud_model.pkl' file was not found. Please train and save the model first.")
    st.stop()

# Set page title and header
st.set_page_config(page_title="Fraud Detection App", layout="centered")
st.title("Fraud Detection App")
st.markdown("Enter transaction details below to predict if it is fraudulent.")

# Create input fields for user data
with st.form("transaction_form"):
    st.header("Transaction Details")
    
    # Use a selectbox for the categorical feature 'type'
    transaction_type = st.selectbox(
        "Transaction Type",
        options=['CASH_OUT', 'PAYMENT', 'CASH_IN', 'TRANSFER', 'DEBIT']
    )
    
    # Use number inputs for numerical features
    amount = st.number_input("Amount", min_value=0.01, format="%.2f")
    oldbalanceOrg = st.number_input("Old Balance (Originator)", min_value=0.0, format="%.2f")
    newbalanceOrig = st.number_input("New Balance (Originator)", min_value=0.0, format="%.2f")
    oldbalanceDest = st.number_input("Old Balance (Destination)", min_value=0.0, format="%.2f")
    newbalanceDest = st.number_input("New Balance (Destination)", min_value=0.0, format="%.2f")

    submitted = st.form_submit_button("Predict")

# When the user clicks the 'Predict' button
if submitted:
    # Create a DataFrame from the user inputs
    input_data = pd.DataFrame([[
        transaction_type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest
    ]], columns=['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest'])
    
    # Make a prediction
    prediction = pipeline.predict(input_data)
    
    # Display the result
    if prediction[0] == 1:
        st.error("This transaction is predicted as **FRAUDULENT**.")
    else:
        st.success("This transaction is predicted as **NOT FRAUDULENT**.")