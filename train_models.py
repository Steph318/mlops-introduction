# Task: train the model from the pipeline

import pickle
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train Logistic Regression model
logreg_model = LogisticRegression(max_iter=200)
logreg_model.fit(X, y)

# Train Random Forest model
rf_model = RandomForestClassifier()
rf_model.fit(X, y)

# Save models to disk
with open("models/logistic_regression.pkl", "wb") as f:
    pickle.dump(logreg_model, f)

with open("models/random_forest.pkl", "wb") as f:
    pickle.dump(rf_model, f)