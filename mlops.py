import streamlit as st
import pandas as pd
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

    # Separate the features and target variable
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Handle missing values
    X.fillna(0, inplace=True)  # Replace missing values with 0

    # Perform one-hot encoding for categorical features
    X_encoded = pd.get_dummies(X)

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
        model = train_linear_regression(X_train, y_train)

        # Text Field
        st.subheader("Carat")
        carat = st.text_input("Enter Carat")
        st.write("Carat:", carat)

        st.subheader("Depth")
        depth = st.text_input("Enter Depth")
        st.write("Depth:", depth)

        st.subheader("Table")
        table = st.text_input("Enter Table")
        st.write("Table:", table)

        st.subheader("x")
        x = st.text_input("Enter x")
        st.write("x:", x)

        st.subheader("y")
        y = st.text_input("Enter y")
        st.write("y:", y)

        st.subheader("z")
        z = st.text_input("Enter z")
        st.write("z:", z)

        # Radio Buttons
        st.subheader("Cut")
        cut = st.radio("Select Cut value", ["Fair", "Good", "Very Good", "Premium", "Ideal"])
        st.write("Cut value:", cut)
        cut_mapping = {"Fair": 1, "Good": 2, "Very Good": 3, "Premium": 4, "Ideal": 5}
        cut_value = cut_mapping[cut]

        st.subheader("Color")
        color = st.radio("Select color", ["D", "E", "F", "G", "H", "I", "J"])
        st.write("Color:", color)
        color_mapping = {"D": 7, "E": 6, "F": 5, "G": 4, "H": 3, "I": 2, "J": 1}
        color_value = color_mapping[color]

        st.subheader("Clarity")
        clarity = st.radio("Select Clarity", ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"])
        st.write("Clarity Level:", clarity)
        clarity_mapping = {"I1": 1, "SI2": 2, "SI1": 3, "VS2": 4, "VS1": 5, "VVS2": 6, "VVS1": 7, "IF": 8}
        clarity_value = clarity_mapping[clarity]

        # Button
        if st.button("Predict Price"):
            st.write("Price Predicted!")
            st.write("volume", float(x) * float(y) * float(z))
            yhat_test = model.predict([[float(carat), cut_value, color_value, clarity_value, float(depth), float(table), float(x) * float(y) * float(z)]])
            st.write("Diamond Price is $", yhat_test[0])

# Run the app
if __name__ == "__main__":
    display_app()
