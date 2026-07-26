
from langchain.prompts import PromptTemplate

# Prompt for explaining prediction
prediction_prompt = PromptTemplate(
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
    template="""
You are an expert AI Car Purchase Advisor.

Customer Details:
- Age: {age}
- Annual Income: {income}
- Credit Score: {credit_score}
- Years Driving: {driving_years}
- Family Size: {family_size}
- Car Ownership: {car_ownership}
- Loan Amount: {loan_amount}

Machine Learning Prediction:
{prediction}

Your task:
1. Explain the prediction in simple English.
2. Explain why the model predicted this result.
3. Give three practical suggestions to improve the chances of purchasing a car if the prediction is "No".
4. Keep the explanation short, friendly, and easy to understand.
"""
)

# Prompt for general chatbot questions
chat_prompt = PromptTemplate(
    input_variables=["question"],
    template="""
You are an AI assistant for a Car Purchase Prediction application.

Answer the following user question clearly and simply.

Question:
{question}

Answer:
"""
)
