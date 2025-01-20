import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load the data
customer = pd.read_csv('https://raw.githubusercontent.com/shreyasnimkhedkar/customer_churn_predition/refs/heads/master/customer_churn.csv')

# Display the dataset
st.write("Customer Churn Dataset")
st.write(customer)

# Display null values in the dataset
st.write("Null Values in Dataset")
st.write(customer.isnull().sum())

# Preprocess the data
customer['Onboard_date'] = pd.to_datetime(customer['Onboard_date'])
customer['Onboard_timestamp'] = customer['Onboard_date'].astype('int64') // 10**9

x = customer.iloc[:, [1, 2, 3, 4]]
y = customer.iloc[:, [9]]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, stratify=y, random_state=2)

# Standardize the data
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Train the logistic regression model
model = LogisticRegression()
model.fit(x_train_scaled, y_train)

# Display the accuracy of the model
train_accuracy = accuracy_score(y_train, model.predict(x_train_scaled))
test_accuracy = accuracy_score(y_test, model.predict(x_test_scaled))
st.write(f"Train Accuracy: {train_accuracy}")
st.write(f"Test Accuracy: {test_accuracy}")

# Streamlit layout
st.title("Customer Churn Prediction")
st.sidebar.title("Navigation")
menu = ["Home", "Predict Churn"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Home":
    st.subheader("Home")
    st.write("Welcome to the Customer Churn Prediction App!")
    st.write(customer)

elif choice == "Predict Churn":
    st.subheader("Predict Customer Churn")
    st.write("Predict whether a customer will convert based on the provided features.")
    
    st.write("Model Details:")
    st.write(f"Train Accuracy: {train_accuracy}")
    st.write(f"Test Accuracy: {test_accuracy}")

    # User inputs
    st.write("Enter customer details for prediction:")
    Age = st.number_input("Age", min_value=0, max_value=100, value=30)
    Gender = st.selectbox("Gender", ["Male", "Female"])
    Income = st.number_input("Income", min_value=0.0, value=50000.0)
    Customer_Type = st.selectbox("Customer Type", ["Type 1", "Type 2", "Type 3"])

    # Encode Gender and Customer Type
    Gender = 1 if Gender == "Male" else 0
    Customer_Type = {"Type 1": 1, "Type 2": 2, "Type 3": 3}[Customer_Type]

    # Create input DataFrame
    input_data = pd.DataFrame({
        'Age': [Age],
        'Gender': [Gender],
        'Income': [Income],
        'Customer_Type': [Customer_Type]
    })

    # Standardize input data
    input_data_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_data_scaled)
    if prediction[0] == 1:
        st.write('Customer will not convert')
    else:
        st.write('Customer will convert')

