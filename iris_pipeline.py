import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from mlflow.models import infer_signature


def load_dataset():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


def train_and_log_model(params):
     X_train, X_test, y_train, y_test = load_dataset()

     model = LogisticRegression(**params)

     with mlflow.start_run():

         model.fit(X_train, y_train)
         accuracy = inference(model, X_test, y_test)

         # Log the hyperparameters
         mlflow.log_params(params)

         # Log the loss metric
         mlflow.log_metric("accuracy", accuracy)

         # Set a tag that we can use to remind ourselves what this run was for
         mlflow.set_tag("Training Info", "Basic LR model for iris data")

         # Infer the model signature
         signature = infer_signature(X_train, model.predict(X_train))

         # Log the model
         model_info = mlflow.sklearn.log_model(
             sk_model=model,
             artifact_path="model",
             signature=signature,
             input_example=X_train,
             registered_model_name="tracking-quickstart",
         )

     return model_info


def inference(model, X_test, y_test):
     predictions = model.predict(X_test)
     accuracy = accuracy_score(y_test, predictions)
     print(f"Accuracy of the model is: {accuracy}.")

     return accuracy


def plot_feature(df, feature):
         # Plot a histogram of one of the features
         df[feature].hist()
         plt.title(f"Distribution of {feature}")
         plt.xlabel(feature)
         plt.ylabel("Frequency")
         plt.show()

def plot_features(df):
    # Plot scatter plot of first two features.
    scatter = plt.scatter(
        df["sepal length (cm)"], df["sepal width (cm)"], c=df["species"]
    )
    plt.title("Scatter plot of the sepal features (width vs length)")
    plt.xlabel(xlabel="sepal length (cm)")
    plt.ylabel(ylabel="sepal width (cm)")
    plt.legend(
        scatter.legend_elements()[0],
        df["species_name"].unique(),
        loc="lower right",
        title="Classes",
    )
    plt.show()

def plot_model(model, X_test, y_test):
    # Plot the confusion matrix for the model
    ConfusionMatrixDisplay.from_estimator(estimator=model, X=X_test, y=y_test)
    plt.title("Confusion Matrix")
    plt.show()



if __name__ == "__main__":
    # Set our tracking server uri for logging
     mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

     # Create a new MLflow Experiment
     mlflow.set_experiment("MLflow Quickstart")

     # Define the model hyperparameters
     params = {
         "solver": "lbfgs",
         "max_iter": 1000,
         "multi_class": "auto",
         "random_state": 8888,
     }
     model_info = train_and_log_model(params)

     X_train, X_test, y_train, y_test = load_dataset()

     # Load the model back for predictions as a generic Python Function model
     loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
     accuracy = inference(loaded_model, X_test, y_test)