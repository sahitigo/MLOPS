# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def train_and_evaluate_model(X_train, y_train, X_test, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    return model, mse

def preprocess_input(input_data):
    # Convert cut, color, and clarity to one-hot encoding
    input_data = pd.get_dummies(input_data, columns=['cut', 'color', 'clarity'], drop_first=True)
    
    return input_data

def main():
    st.title("Diamond Price Estimator")
    st.write("Welcome to the Diamond Price Estimator app!")

    # Load the diamond dataset
    diamonds_data = {
        'carat': [0.23, 0.21, 0.23, 0.29, 0.31],
        'cut': ['Ideal', 'Premium', 'Good', 'Premium', 'Good'],
        'color': ['E', 'E', 'E', 'I', 'J'],
        'clarity': ['SI2', 'SI1', 'VS1', 'VS2', 'SI2'],
        'depth': [61.5, 59.8, 56.9, 62.4, 63.3],
        'table': [55.0, 61.0, 65.0, 58.0, 58.0],
        'price': [326, 326, 327, 334, 335],
        'x': [3.95, 3.89, 4.05, 4.20, 4.34],
        'y': [3.98, 3.84, 4.07, 4.23, 4.35],
        'z': [2.43, 2.31, 2.31, 2.63, 2.75]
    }
    diamonds = pd.DataFrame(diamonds_data)

    # Preprocess the data
    X = diamonds.drop('price', axis=1)
    y = diamonds['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert categorical features to one-hot encoding
    X_train = preprocess_input(X_train)
    X_test = preprocess_input(X_test)

    # Train the model
    model, mse = train_and_evaluate_model(X_train, y_train, X_test, y_test)
        
    st.write("Model trained successfully!")
    st.write("Evaluation Results:")
    st.write(f"Mean Squared Error: {mse}")

    # User input for prediction
    st.write("Make a Price Prediction")
    carat = st.number_input("Carat:")
    cut = st.selectbox("Cut:", ['Premium', 'Good'])
    color = st.selectbox("Color:", ['E', 'I', 'J'])
    clarity = st.selectbox("Clarity:", ['SI1', 'VS2'])
    depth = st.number_input("Depth:")
    table = st.number_input("Table:")
    x = st.number_input("x:")
    y = st.number_input("y:")
    z = st.number_input("z:")

    # Preprocess the input
    input_data = pd.DataFrame([[carat, cut, color, clarity, depth, table, x, y, z]],
                              columns=['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z'])
    input_data = preprocess_input(input_data)

    # Align the input data columns with the training data columns
    input_data = input_data.reindex(columns=X_train.columns, fill_value=0)

    # Make prediction
    prediction = model.predict(input_data)[0]

    st.write("Predicted Price:", prediction)

if __name__ == '__main__':
    main()
