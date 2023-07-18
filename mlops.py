# app.py
import streamlit as st
import seaborn as sns
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
    
    return accuracy, f1, sensitivity

def main():
    st.title("Diamond Classifier")
    st.write("Welcome to the Diamond Classifier app!")

    # Load the diamond dataset
    diamonds = sns.load_dataset('diamonds')

    # Preprocess the data
    X = diamonds.drop('cut', axis=1)
    y = diamonds['cut']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train and evaluate the model
    if st.button("Train Model"):
        accuracy, f1, sensitivity = train_and_evaluate_model(X_train, y_train, X_test, y_test)
        
        st.write("Model trained successfully!")
        st.write("Evaluation Results:")
        st.write(f"Accuracy: {accuracy}")
        st.write(f"F1 Score: {f1}")
        st.write(f"Sensitivity: {sensitivity}")

if __name__ == '__main__':
    main()
