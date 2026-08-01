import os
import streamlit as st
import pandas as pd
import joblib

# Load the model committed by the pipeline (sits next to this file)
model_path = os.path.join(os.path.dirname(__file__), "best_tourism_model.joblib")
model = joblib.load(model_path)

st.title("Tourism Package Prediction App")
st.write("""
This application predicts whether a customer is likely to purchase a tourism package 
based on their demographic and financial details. It loads the model that the GitHub 
Actions pipeline trained and committed into the repo, collects user inputs, makes 
predictions, and displays the results in a simple interface. Streamlit Community Cloud 
runs this file directly from the repo, so there's nothing to configure beyond pointing 
it at tourism_project/deployment/app.py when you deploy.
""")

# Collect user inputs
age = st.number_input("Age", min_value=18, max_value=100, value=30)
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
income = st.number_input("Annual Income (in USD)", min_value=1000, max_value=200000, value=50000)

# Create input dataframe
input_data = pd.DataFrame([{
    "Age": age,
    "Gender": gender,
    "Income": income
}])

# Prediction
if st.button("Predict Package Purchase"):
    prediction = model.predict(input_data)[0]
    result = "Likely to Purchase" if prediction == 1 else "Not Likely to Purchase"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
