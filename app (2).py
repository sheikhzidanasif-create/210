
import streamlit as st
import pandas as pd
import joblib

# Import AI Agent
from ai_agent import explain_prediction

# ===========================
# Load Model
# ===========================
model = joblib.load("car_purchase_model.pkl")

st.set_page_config(page_title="Car Purchase Prediction", page_icon="🚗")

st.title("🚗 Car Purchase Prediction AI Agent")
st.write("Enter customer details to predict whether they are likely to purchase a car.")

# ===========================
# User Inputs
# ===========================

age = st.number_input("Age", 18, 100, 30)

income = st.number_input(
    "Annual Income",
    min_value=10000,
    max_value=500000,
    value=50000,
)

credit_score = st.number_input(
    "Credit Score",
    min_value=300,
    max_value=850,
    value=700,
)

driving_years = st.number_input(
    "Years Driving",
    min_value=0,
    max_value=50,
    value=5,
)

family_size = st.number_input(
    "Family Size",
    min_value=1,
    max_value=10,
    value=4,
)

car_ownership = st.selectbox(
    "Already Own a Car?",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No",
)

loan_amount = st.number_input(
    "Loan Amount",
    min_value=0,
    max_value=100000,
    value=15000,
)

# ===========================
# Prediction
# ===========================

if st.button("Predict"):

    input_data = pd.DataFrame({
        "Age": [age],
        "Annual_Income": [income],
        "Credit_Score": [credit_score],
        "Years_Driving": [driving_years],
        "Family_Size": [family_size],
        "Car_Ownership": [car_ownership],
        "Loan_Amount": [loan_amount]
    })

    prediction = model.predict(input_data)[0]

    result = "Yes" if prediction == 1 else "No"

    st.subheader("Prediction")

    if prediction == 1:
        st.success("Customer is likely to purchase a car.")
    else:
        st.error("Customer is unlikely to purchase a car.")

    st.write(f"**Prediction:** {result}")

    st.divider()

    st.subheader("AI Explanation")

    explanation = explain_prediction(
        age=age,
        income=income,
        credit_score=credit_score,
        driving_years=driving_years,
        family_size=family_size,
        car_ownership=car_ownership,
        loan_amount=loan_amount,
        prediction=result
    )

    st.write(explanation)
