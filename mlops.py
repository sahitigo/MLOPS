import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
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

    # Separate the features and target variable
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Handle missing values
    X.fillna(0, inplace=True)  # Replace missing values with 0

    # Handle outliers in the target variable (y)
    Q1 = y.quantile(0.25)
    Q3 = y.quantile(0.75)
    IQR = Q3 - Q1
    y_outlier_removed = y[(y >= Q1 - 1.5 * IQR) & (y <= Q3 + 1.5 * IQR)]

    # Remove rows with null values in both features and target variable
    df_cleaned = df.dropna(subset=[target_column] + list(X.columns))

    # Separate the cleaned features and target variable
    X_cleaned = df_cleaned.drop(columns=[target_column])
    y_cleaned = df_cleaned[target_column]

    # Perform one-hot encoding for categorical features
    X_encoded = pd.get_dummies(X_cleaned)

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_cleaned, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


# Train a linear regression model
def train_decision_tree_regression(X, y):
    model = DecisionTreeRegressor(random_state=42)
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
        X_train, X_test, y_train, y_test = process_file(uploaded_file, target_column)

        # Check if any of the variables are None (indicating an error occurred)
        if X_train is None or X_test is None or y_train is None or y_test is None:
            return

        # Train the linear regression model (call the train_linear_regression function)
         model = train_decision_tree_regression(X_train, y_train)

        # Get user inputs for feature values
        feature_values = {}
        for feature in X_train.columns:
            value = st.text_input(f"Enter value for {feature}")
            feature_values[feature] = float(value) if value else None

        # Create a DataFrame with the user inputs
        input_df = pd.DataFrame([feature_values])

        # Perform one-hot encoding for the input data
        input_encoded = pd.get_dummies(input_df)

        # Align input data with training data to ensure consistent columns
        input_encoded = input_encoded.reindex(columns=X_train.columns, fill_value=0)

        # Handle missing values in the user input
        input_encoded.fillna(0, inplace=True)  # Replace missing values with 0

        # Make predictions on the input data
        y_pred = model.predict(input_encoded)

        # Display the predictions
        st.subheader("Prediction")
        st.write("Target Column:", target_column)
        st.write("Predicted Value:", y_pred[0])

        # Calculate MAPE for training and test sets
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        train_mape = calculate_mape(y_train, y_train_pred)
        test_mape = calculate_mape(y_test, y_test_pred)

        st.subheader("Model Performance")
        st.write("Training MAPE:", train_mape)
        st.write("Test MAPE:", test_mape)

# Run the app
if __name__ == "__main__":
    display_app()
