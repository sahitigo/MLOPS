import streamlit as st
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression

# Load the California Housing dataset
data = fetch_california_housing()
X = data.data
y = data.target

# Train a linear regression model
model = LinearRegression()
model.fit(X, y)

# Display input fields for feature values
st.title("California Housing Price Prediction")
st.header("Enter Feature Values")
feature_names = data.feature_names
feature_values = []

for feature_name in feature_names:
    value = st.text_input(feature_name)
    feature_values.append(float(value) if value else 0.0)

# Predict the housing price based on the input features
prediction = model.predict([feature_values])[0]

# Display the predicted housing price
st.header("Predicted Housing Price")
st.write(f"${prediction:.2f}")
