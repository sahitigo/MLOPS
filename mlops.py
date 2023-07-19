import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

# Function to process the uploaded file
def process_file(upload_file, target_column):
    df = pd.read_csv(upload_file)
    if target_column not in df.columns:
        st.error(f"Target column '{target_column}' not found in the dataset.")
        return None, None, None, None

    X = df.drop(columns=[target_column])  # Drop the target column from features
    y = df[target_column]  # Set the target column as the target variable

    # Perform one-hot encoding for categorical features
    categorical_features = X.select_dtypes(include=['object']).columns
    X_encoded = pd.get_dummies(X, columns=categorical_features)

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

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

        # Process the uploaded file
        X_train, X_test, y_train, y_test = process_file(uploaded_file, target_column)

        if X_train is None or X_test is None or y_train is None or y_test is None:
            return

        # Train the linear regression model
        model = train_linear_regression(X_train, y_train)

        # Predict on the training set
        y_train_pred = model.predict(X_train)
        mape_train = calculate_mape(y_train, y_train_pred)

        # Predict on the test set
        y_test_pred = model.predict(X_test)
        mape_test = calculate_mape(y_test, y_test_pred)

        # Display the results
        st.subheader("Data Summary")
        st.write("Train set samples:", X_train.shape[0])
        st.write("Test set samples:", X_test.shape[0])
        st.write("Number of features:", X_train.shape[1])
        st.subheader("Model Evaluation")
        st.write("Train set MAPE:", mape_train)
        st.write("Test set MAPE:", mape_test)

# Run the app
if __name__ == "__main__":
    display_app()
