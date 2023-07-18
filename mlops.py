import seaborn as sns
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.ensemble import RandomForestClassifier

# Load the diamond dataset
diamonds = sns.load_dataset('diamonds')

# Preprocess the data
X = diamonds.drop('cut', axis=1)
y = diamonds['cut']

# Encode the categorical target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the column transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), ['color', 'clarity']),
        ('num', StandardScaler(), ['carat', 'depth', 'table', 'x', 'y', 'z'])
    ])

# Preprocess the training data
X_train = preprocessor.fit_transform(X_train)

# Preprocess the testing data
X_test = preprocessor.transform(X_test)

# Define the random forest classifier
clf = RandomForestClassifier(random_state=42)

# Define the hyperparameters for grid search
params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10]
}

# Perform grid search to find the best hyperparameters
grid_search = GridSearchCV(clf, params, cv=3)
grid_search.fit(X_train, y_train)

# Get the best model and its predictions
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
sensitivity = recall_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")
print(f"Sensitivity: {sensitivity}")