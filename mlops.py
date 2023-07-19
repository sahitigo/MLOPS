import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error

# Function to process the uploaded file
def process_file(upload_file, target_column):
    df = pd.read_csv(upload_file)
    X = df.drop(columns=[target_column])  # Drop the target column from features
    y = df[target_column]  # Set the target column as the target variable

    # Perform one-hot encoding for categorical features
    categorical_features = X.select_dtypes(include=['object']).columns
    X_encoded = pd.get_dummies(X, columns=categorical_features)

    return X_encoded, y

# Train a linear regression model
def train_linear_regression(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

# Function to calculate MAPE
def calculate_mape(y_true, y_pred):
    return mean_absolute_percentage_error(y_true, y_pred) * 100

# Display file upload and model evaluation
def display_app():
    st.title("Linear Regression with Custom Dataset")
    st.header("Upload Your Data")

    # File upload control
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Prompt user to enter target column
        target_column = st.text_input("Enter the target column name")

        # Process the uploaded file
        X, y = process_file(uploaded_file, target_column)

        # Train the linear regression model
        model = train_linear_regression(X, y)

        # Predict on the training set
        y_pred = model.predict(X)

        # Calculate MAPE
        mape = calculate_mape(y, y_pred)

        # Display the results
        st.subheader("Data Summary")
        st.write("Number of samples:", X.shape[0])
        st.write("Number of features:", X.shape[1])
        st.subheader("Model Evaluation")
        st.write("Mean Absolute Percentage Error (MAPE):", mape)

# Run the app
if __name__ == "__main__":
    display_app()
