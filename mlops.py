# app.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split

def train_and_evaluate_model(X_train, y_train, X_test, y_test):
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    sensitivity = recall_score(y_test, y_pred, average='weighted')
    
    return clf, accuracy, f1, sensitivity

def main():
    st.title("Diamond Classifier")
    st.write("Welcome to the Diamond Classifier app!")

    # Load the diamond dataset
    diamonds = sns.load_dataset('diamonds').copy()

    # Preprocess the data
    X = diamonds.drop('cut', axis=1)
    y = diamonds['cut']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model, accuracy, f1, sensitivity = train_and_evaluate_model(X_train, y_train, X_test, y_test)
        
    st.write("Model trained successfully!")
    st.write("Evaluation Results:")
    st.write(f"Accuracy: {accuracy}")
    st.write(f"F1 Score: {f1}")
    st.write(f"Sensitivity: {sensitivity}")

    # Visualize feature importances
    importance = model.feature_importances_
    feature_names = X.columns
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
    feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)

    st.write("Feature Importances:")
    fig, ax = plt.subplots()
    sns.barplot(data=feature_importance_df, x='Importance', y='Feature', ax=ax)
    ax.set_title("Feature Importances")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    st.pyplot(fig)

    # User input for prediction
    st.write("Make a Prediction")
    carat = st.number_input("Carat:")
    depth = st.number_input("Depth:")
    table = st.number_input("Table:")
    x = st.number_input("x:")
    y = st.number_input("y:")
    z = st.number_input("z:")
    color = st.selectbox("Color:", ['D', 'E', 'F', 'G', 'H', 'I', 'J'])
    clarity = st.selectbox("Clarity:", ['IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2', 'I1'])

    # Preprocess the input
    input_data = [[carat, depth, table, x, y, z, color, clarity]]
    input_df = pd.DataFrame(input_data, columns=['carat', 'depth', 'table', 'x', 'y', 'z', 'color', 'clarity'])

    # Make prediction
    prediction = None
    if carat and depth and table and x and y and z and color and clarity:
        prediction = model.predict(input_df)[0]

    if prediction is not None:
        st.write("Prediction:", prediction)

if __name__ == '__main__':
    main()
