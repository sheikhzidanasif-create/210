
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# =====================================
# Set Your Gemini API Key
# =====================================
os.environ["GOOGLE_API_KEY"] = "YOUR_GEMINI_API_KEY"

# =====================================
# Load Gemini Model
# =====================================
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.3
)

# =====================================
# Prompt Template
# =====================================
template = """
You are an AI Car Purchase Advisor.

Customer Information:
- Age: {age}
- Annual Income: {income}
- Credit Score: {credit_score}
- Years Driving: {driving_years}
- Family Size: {family_size}
- Car Ownership: {car_ownership}
- Loan Amount: {loan_amount}

The machine learning model predicted:
Prediction: {prediction}

Explain:
1. Why this prediction was made.
2. Whether the customer is likely to purchase a car.
3. Give 3 recommendations to improve eligibility if prediction is No.
4. Keep the explanation simple.
"""

prompt = PromptTemplate(
    input_variables=[
        "age",
        "income",
        "credit_score",
        "driving_years",
        "family_size",
        "car_ownership",
        "loan_amount",
        "prediction"
    ],
    template=template
)

chain = LLMChain(
    llm=llm,
    prompt=prompt
)

# =====================================
# Function
# =====================================
def explain_prediction(
    age,
    income,
    credit_score,
    driving_years,
    family_size,
    car_ownership,
    loan_amount,
    prediction
):
    response = chain.run(
        age=age,
        income=income,
        credit_score=credit_score,
        driving_years=driving_years,
        family_size=family_size,
        car_ownership=car_ownership,
        loan_amount=loan_amount,
        prediction=prediction
    )
    return response


# =====================================
# Example
# =====================================
if __name__ == "__main__":

    result = explain_prediction(
        age=30,
        income=60000,
        credit_score=740,
        driving_years=10,
        family_size=4,
        car_ownership=1,
        loan_amount=15000,
        prediction="Yes"
    )

    print(result)
