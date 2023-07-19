import streamlit as st
import pandas as pd
import statsmodels.api as sm
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

# Train a decision tree regression model
def train_decision_tree_regression(X, y):
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X, y)
    return model

# Function to calculate MAPE
def calculate_mape(y_true, y_pred):
    return mean_absolute_percentage_error(y_true, y_pred) * 100

# Function to get input from user (text box for numeric, radio button for categorical)
def get_user_input(feature, is_numeric):
    if is_numeric:
        value = st.text_input(f"Enter value for {feature}")
        return float(value) if value else None
    else:
        unique_values = feature_values[feature]
        selected_value = st.radio(f"Select value for {feature}", unique_values)
        return selected_value

# Function to perform backward elimination
def backward_elimination(X, y):
    cols = list(X.columns)
    p_values = pd.Series(dtype='float64')
    
    while len(cols) > 0:
        Xc = X[cols]
        model = sm.OLS(y, Xc).fit()
        p = model.pvalues
        pmax = p.max()
        pid = p.idxmax()
        p_values = p_values.append(pd.Series({pid: pmax}))
        
        if pmax > 0.05:
            cols.remove(pid)
            print('Variable removed:', pid, 'P-value:', pmax)
        else:
            break

    return cols, p_values

# Display file upload and model evaluation
def display_app():
    st.title("Decision Tree Regression with Diamond Dataset")
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

        # Perform backward elimination
        selected_cols, p_values = backward_elimination(X_train, y_train)

        # Train the decision tree regression model (call the train_decision_tree_regression function)
        model = train_decision_tree_regression(X_train[selected_cols], y_train)

        # Get user inputs for feature values
        feature_values = {}
        for feature in selected_cols:
            is_numeric = X_train[feature].dtype.kind in 'biufc'  # Check if the feature is numeric
            feature_values[feature] = get_user_input(feature, is_numeric)

        # Create a DataFrame with the user inputs
        input_df = pd.DataFrame([feature_values])

        # Perform one-hot encoding for the input data
        input_encoded = pd.get_dummies(input_df)

        # Align input data with training data to ensure consistent columns
        input_encoded = input_encoded.reindex(columns=selected_cols, fill_value=0)

        # Handle missing values in the user input
        input_encoded.fillna(0, inplace=True)  # Replace missing values with 0

        # Make predictions on the input data
        y_pred = model.predict(input_encoded)

        # Display the predictions
        st.subheader("Prediction")
        st.write("Target Column:", target_column)
        st.write("Predicted Value:", y_pred[0])

        # Calculate MAPE for training and test sets
        y_train_pred = model.predict(X_train[selected_cols])
        y_test_pred = model.predict(X_test[selected_cols])
        train_mape = calculate_mape(y_train, y_train_pred)
        test_mape = calculate_mape(y_test, y_test_pred)

        st.subheader("Model Performance")
        st.write("Training MAPE:", train_mape)
        st.write("Test MAPE:", test_mape)

        # Display the variables and their corresponding p-values after backward elimination
        st.subheader("Backward Elimination Results")
        p_values.index = selected_cols
        st.write("Selected Variables:", selected_cols)
        st.write("P-values:", p_values)

# Run the app
if __name__ == "__main__":
    display_app()
