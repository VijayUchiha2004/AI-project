# app.py
import streamlit as st
import pandas as pd
from model import load_data, train_model

# Load and train model
data = load_data()
model, vectorizer, accuracy = train_model(data)

# Streamlit UI
st.title("📊 AI-Powered Personal Finance Tracker")

uploaded_file = st.file_uploader("Upload Expense File (CSV)", type=["csv"])
if uploaded_file:
    new_data = pd.read_csv(uploaded_file)
    new_data['Description'] = new_data['Description'].str.lower()
    X_new = vectorizer.transform(new_data['Description'])
    new_data['Predicted Category'] = model.predict(X_new)

    st.subheader("Categorized Expenses")
    st.dataframe(new_data)

    # Budget Summary
    category_summary = new_data.groupby('Predicted Category')['Amount'].sum().reset_index()
    st.subheader("Budget Summary")
    st.bar_chart(category_summary.set_index('Predicted Category'))

# Show model accuracy
if st.checkbox("Show Model Accuracy"):
    st.write(f"Model Accuracy: {accuracy * 100:.2f}%")
