import pandas as pd
import pickle
import mlflow
import mlflow.sklearn
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def main():
    # 1. Load the dataset
    # We will try to load from a local file, otherwise fallback to the UCI repository URL
    local_path = "winequality-red.csv"
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    
    if os.path.exists(local_path):
        print(f"Loading data from {local_path}...")
        df = pd.read_csv(local_path, sep=';')
    else:
        print(f"Local file not found. Downloading data from {url}...")
        df = pd.read_csv(url, sep=';')

    # 2. Preprocessing
    # Separating features and target variable
    X = df.drop('quality', axis=1)
    y = df['quality']

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Setup MLflow Experiment
    mlflow.set_experiment("WineQuality_RandomForest_Exp")

    with mlflow.start_run():
        # Define hyperparameters
        n_estimators = 100
        max_depth = 10
        random_state = 42

        # 4. Train the Model
        print("Training Random Forest Classifier...")
        rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        rf.fit(X_train, y_train)

        # 5. Evaluate the Model
        predictions = rf.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"Model Accuracy: {accuracy:.4f}")

        # 6. Log Parameters and Metrics to MLflow
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", accuracy)
        
        # Log the model artifact to MLflow
        mlflow.sklearn.log_model(rf, "random_forest_model")
        print("Experiment logged to MLflow.")

        # 7. Save Model as Pickle File
        pickle_filename = "model.pkl"
        with open(pickle_filename, 'wb') as f:
            pickle.dump(rf, f)
        print(f"Model saved locally as '{pickle_filename}'.")

        # 8. Load the Artifacts (Pickle)
        print("Loading artifacts to verify...")
        with open(pickle_filename, 'rb') as f:
            loaded_model = pickle.load(f)
        
        # Verify loaded model works
        check_pred = loaded_model.predict(X_test[:5])
        print(f"Verification - Predictions from loaded model: {check_pred}")

if __name__ == "__main__":
    main()
