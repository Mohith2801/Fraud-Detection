import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Load the trained model and data
@st.cache_resource
def load_data_and_model():
    """Load the dataset and the trained model."""
    try:
        # CORRECT: Use relative paths for files in the same repository
        data = pd.read_csv("Fraud.csv")
        model = joblib.load("fraud_model.pkl")
        return data, model
    except FileNotFoundError:
        st.error("Error: The 'Fraud.csv' or 'fraud_model.pkl' file was not found. Please ensure both files are in the same GitHub repository as your app.py file.")
        return None, None

data, model = load_data_and_model()

if data is not None and model is not None:
    # Set the title of the Streamlit app
    st.title("Fraud Detection App")

    # Display some information about the dataset
    st.header("Dataset Overview")
    st.write(data.head())
    
    # User input for new transaction
    st.header("Predict Fraud for a New Transaction")
    
    # Create input fields for features
    step = st.number_input('Step', min_value=0)
    type_trans = st.selectbox('Type', data['type'].unique())
    amount = st.number_input('Amount', min_value=0.0)
    nameOrig = st.text_input('Name of Originator (nameOrig)')
    oldbalanceOrg = st.number_input('Old Balance Originator', min_value=0.0)
    newbalanceOrig = st.number_input('New Balance Originator', min_value=0.0)
    nameDest = st.text_input('Name of Destination (nameDest)')
    oldbalanceDest = st.number_input('Old Balance Destination', min_value=0.0)
    newbalanceDest = st.number_input('New Balance Destination', min_value=0.0)
    isFlaggedFraud = st.selectbox('isFlaggedFraud', [0, 1])

    if st.button('Predict'):
        # Prepare the input for prediction
        input_data = pd.DataFrame([[step, type_trans, amount, nameOrig, oldbalanceOrg,
                                      newbalanceOrig, nameDest, oldbalanceDest,
                                      newbalanceDest, isFlaggedFraud]],
                                  columns=['step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg',
                                           'newbalanceOrig', 'nameDest', 'oldbalanceDest',
                                           'isFlaggedFraud'])
        
        # Make a prediction
        prediction = model.predict(input_data)
        
        # Display the result
        st.subheader("Prediction Result:")
        if prediction[0] == 1:
            st.error("Fraudulent Transaction")
        else:
            st.success("Non-Fraudulent Transaction")
