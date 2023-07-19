import streamlit as st
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error

# Function to process the uploaded file
def process_file(upload_file):
    df = pd.read_csv(upload_file)
    X = df.iloc[:, :-1]  # Assuming the features are in the first columns
    y = df.iloc[:, -1]   # Assuming the target variable is in the last column

    # Perform one-hot encoding for categorical features
    categorical_features = X.select_dtypes(include=['object']).columns
    ct = ColumnTransformer(
        transformers=[('encoder', OneHotEncoder(), categorical_features)],
        remainder='passthrough'
    )
    X_encoded = ct.fit_transform(X)

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
        # Process the uploaded file
        X, y = process_file(uploaded_file)

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
