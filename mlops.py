# app.py
import streamlit as st
import seaborn as sns
import pandas as pd
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
    # Convert color to one-hot encoding
    input_data = pd.get_dummies(input_data, columns=['color'], drop_first=True)
    
    # Convert cut to one-hot encoding and drop one category to avoid multicollinearity
    input_data = pd.get_dummies(input_data, columns=['cut'], drop_first=True)
    
    return input_data

def main():
    st.title("Diamond Price Estimator")
    st.write("Welcome to the Diamond Price Estimator app!")

    # Load the diamond dataset from seaborn
    diamonds = sns.load_dataset('diamonds')

    # Preprocess the data
    X = diamonds.drop('price', axis=1)
    y = diamonds['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model, mse = train_and_evaluate_model(X_train, y_train, X_test, y_test)
        
    st.write("Model trained successfully!")
    st.write("Evaluation Results:")
    st.write(f"Mean Squared Error: {mse}")

    # User input for prediction
    st.write("Make a Price Prediction")
    carat = st.number_input("Carat:")
    cut = st.selectbox("Cut:", ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'])
    color = st.selectbox("Color:", ['D', 'E', 'F', 'G', 'H', 'I', 'J'])
    clarity = st.selectbox("Clarity:", ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'])
    depth = st.number_input("Depth:")
    table = st.number_input("Table:")
    x = st.number_input("x:")
    y = st.number_input("y:")
    z = st.number_input("z:")

    # Preprocess the input
    input_data = pd.DataFrame([[carat, cut, color, clarity, depth, table, x, y, z]],
                              columns=['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z'])
    input_data = preprocess_input(input_data)

    # Make prediction
    prediction = None
    if not input_data.empty:
        prediction = model.predict(input_data)[0]

    if prediction is not None:
        st.write("Predicted Price:", prediction)

if __name__ == '__main__':
    main()
