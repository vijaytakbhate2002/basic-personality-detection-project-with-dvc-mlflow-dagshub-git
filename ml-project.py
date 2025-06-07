import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
import os
warnings.filterwarnings('ignore')

# MLFLOW_TRACKING_URI = "ADD YOUR TRACKING URI HERE"
# mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# os.environ["MLFLOW_TRACKING_USERNAME"] = "ADD YOUR USERNAME HERE"
# os.environ["MLFLOW_TRACKING_PASSWORD"] = "ADD YOUR PASSWORD HERE"


# Create experiments directory if it doesn't exist
if not os.path.exists("mlruns"):
    os.makedirs("mlruns")

# Set MLflow experiment name
try:
    mlflow.set_experiment("Personality Prediction")
except Exception as e:
    print(f"Error setting experiment: {e}")
    # Create new experiment if it doesn't exist
    mlflow.create_experiment("Personality Prediction")
    mlflow.set_experiment("Personality Prediction")

# Load and preprocess the data
def load_data(file_path):
    df = pd.read_csv(file_path)
    
    # Convert categorical variables
    le = LabelEncoder()
    df['Stage_fear'] = le.fit_transform(df['Stage_fear'])
    df['Drained_after_socializing'] = le.fit_transform(df['Drained_after_socializing'])
    df['Personality'] = le.fit_transform(df['Personality'])
    df = df.fillna(df.mean())
    
    return df

try:
    # Start MLflow run
    with mlflow.start_run(run_name="Personality Prediction Model") as run:
        print(f"Started MLflow run with ID: {run.info.run_id}")
        
        # Load data
        df = load_data("../6-Dagshub_with_dvc/data/personality_dataset.csv")
        
        # Prepare features and target
        X = df.drop('Personality', axis=1)
        y = df['Personality']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Define model parameters
        params = {
            "n_estimators": 100,
            "max_depth": 5,
            "random_state": 42
        }
        
        # Log parameters
        mlflow.log_params(params)
        print("Logged parameters successfully")
        
        # Train the model
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        print("Logged metrics successfully")
        
        # Log feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        })
        feature_importance.to_csv('feature_importance.csv', index=False)
        mlflow.log_artifact('feature_importance.csv')
        print("Logged feature importance successfully")
        
        # Log the model
        mlflow.sklearn.log_model(model, "personality_model")
        print("Logged model successfully")
        
        # Print results
        print("\nModel Performance Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        print("\nFeature Importance:")
        print(feature_importance.sort_values('importance', ascending=False))
        
        print("\nMLflow run completed successfully!")
        print(f"Run ID: {run.info.run_id}")
        print("You can view the results in the MLflow UI at http://localhost:5000")

except Exception as e:
    print(f"An error occurred: {e}")
    raise