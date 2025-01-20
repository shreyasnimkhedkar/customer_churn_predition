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

# Preprocess the data (exclude Onboard_date)
x = customer[['Account_Manager', 'Total_Purchase', 'Years', 'Num_Sites']]
y = customer['Churn']

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
    st.write("Predict whether a customer will churn based on the provided features.")
    
    st.write("Model Details:")
    st.write(f"Train Accuracy: {train_accuracy}")
    st.write(f"Test Accuracy: {test_accuracy}")

    st.write("Enter customer details for prediction:")
    Account_Manager = st.selectbox("Account Manager", [0, 1])
    Total_Purchase = st.number_input("Total Purchase", min_value=0.0, value=10000.0)
    Years = st.number_input("Years", min_value=0, max_value=10, value=5)
    Num_Sites = st.number_input("Number of Sites", min_value=0, max_value=100, value=5)

    input_data = pd.DataFrame({
        'Account_Manager': [Account_Manager],
        'Total_Purchase': [Total_Purchase],
        'Years': [Years],
        'Num_Sites': [Num_Sites]
    })

    input_data_scaled = scaler.transform(input_data)

    prediction = model.predict(input_data_scaled)
    if prediction[0] == 1:
        st.write('Customer will churn')
    else:
        st.write('Customer will not churn')
