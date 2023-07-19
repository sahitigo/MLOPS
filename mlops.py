import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

# Function to process the uploaded file
def process_file(upload_file, target_column):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(upload_file)

    # Check if the target column exists in the DataFrame
    if target_column not in df.columns:
        st.error(f"Target column '{target_column}' not found in the dataset.")
        return None, None, None, None

    # Separate numeric and categorical columns
    numeric_features = df.select_dtypes(include='number').columns
    categorical_features = df.select_dtypes(include='object').columns

    # Drop the target column from features and set it as the target variable
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert numeric features to float
    X_train[numeric_features] = X_train[numeric_features].astype(float)
    X_test[numeric_features] = X_test[numeric_features].astype(float)

    return X_train, X_test, y_train, y_test, numeric_features, categorical_features

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
    st.title("Linear Regression with Diamond Dataset")
    st.header("Upload Your Data")

    # File upload control
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Prompt user to enter target column
        target_column = st.text_input("Enter the target column name")

        # Process the uploaded file (call the process_file function)
        X_train, X_test, y_train, y_test, numeric_features, categorical_features = process_file(uploaded_file, target_column)

        # Check if any of the variables are None (indicating an error occurred)
        if X_train is None or X_test is None or y_train is None or y_test is None:
            return

        # Train the linear regression model (call the train_linear_regression function)
        model = train_linear_regression(X_train, y_train)

        # Get user inputs for feature values (numeric columns)
        numeric_values = {}
        for feature in numeric_features:
            value = st.number_input(f"Enter value for {feature}")
            numeric_values[feature] = float(value) if value else np.nan

        # Get user inputs for feature values (categorical columns)
        categorical_values = {}
        for feature in categorical_features:
            value = st.text_input(f"Enter value for {feature}")
            categorical_values[feature] = value

        # Create a DataFrame with the user inputs
        input_df = pd.DataFrame([numeric_values])
        input_df = pd.concat([input_df, pd.DataFrame([categorical_values])], axis=1)

        # Convert numeric features to float
        input_df[numeric_features] = input_df[numeric_features].astype(float)

        # Perform one-hot encoding for categorical features
        input_encoded = pd.get_dummies(input_df, columns=categorical_features)

        # Ensure input DataFrame has the same columns as training data
        input_encoded = input_encoded.reindex(columns=X_train.columns, fill_value=0)

        # Make predictions on the input data
        y_pred = model.predict(input_encoded)

        # Display the predictions
        st.subheader("Prediction")
        st.write("Target Column:", target_column)
        st.write("Predicted Value:", y_pred[0])

# Run the app
if __name__ == "__main__":
    display_app()
