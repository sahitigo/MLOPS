import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error

# Function to process the uploaded file
def process_file(upload_file, target_column, drop_columns):
    df = pd.read_csv(upload_file)
    df = df.drop(columns=drop_columns)  # Drop the specified columns
    X = df.drop(columns=[target_column])  # Drop the target column from features
    y = df[target_column]  # Set the target column as the target variable
    return X, y

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

        # Prompt user to enter columns to drop
        drop_columns_input = st.text_input("Enter columns to drop (comma-separated)")

        # Split the input into individual column names
        drop_columns = [col.strip() for col in drop_columns_input.split(",")]

        # Process the uploaded file
        X, y = process_file(uploaded_file, target_column, drop_columns)

        # Get user inputs for feature values
        feature_values = {}
        for feature in X.columns:
            value = st.text_input(f"Enter value for {feature}")
            feature_values[feature] = value

        # Create a DataFrame with the user inputs
        input_df = pd.DataFrame([feature_values])

        # Train the linear regression model using the original features
        model = train_linear_regression(X, y)

        # Perform one-hot encoding for categorical features
        X_encoded = pd.get_dummies(X, columns=X.select_dtypes(include=['object']).columns)

        # Ensure input DataFrame has the same columns as training data
        input_encoded = pd.get_dummies(input_df, columns=X.select_dtypes(include=['object']).columns)
        input_encoded = input_encoded.reindex(columns=X_encoded.columns, fill_value=0)

        # Make predictions on the input data
        y_pred = model.predict(input_encoded)

        # Display the predictions
        st.subheader("Prediction")
        st.write("Target Column:", target_column)
        st.write("Predicted Value:", y_pred[0])

# Run the app
if __name__ == "__main__":
    display_app()
