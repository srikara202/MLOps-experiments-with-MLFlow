import os
import joblib
import pandas as pd
import shutil
from mlflow.models.signature import infer_signature
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub

dagshub.init(repo_owner='srikara202', repo_name='MLOps-experiments-with-MLFlow', mlflow=True)

mlflow.set_tracking_uri('https://dagshub.com/srikara202/MLOps-experiments-with-MLFlow.mlflow')

# load wine dataset
wine = load_wine()
X = wine.data
y = wine.target

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# Define the params for RF model
max_depth = 8
n_estimators = 6

# Mention your experiment below
mlflow.set_experiment('MLOPS-Exp-2')

with mlflow.start_run():
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(X_train,y_train)

    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_param('n_estimators', n_estimators)

    # creating a confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')

    # save plot
    plt.savefig('Confusion-matrix.png')

    # log artifacts using mlflow
    mlflow.log_artifact("Confusion-matrix.png")
    mlflow.log_artifact(__file__)

    # tags
    mlflow.set_tags({'Author':'Srikara','Project':'Wine Quality Prediction'})

    # log the model

    # get current run_id for uniquing
    run_id = mlflow.active_run().info.run_id

    # 1) Paths for artifacts
    pickle_path = f"models/random_forest_md{max_depth}_ne{n_estimators}_{run_id}.pkl"
    mlflow_model_dir = f"models/mlflow_rf_model_md{max_depth}_ne{n_estimators}_{run_id}"

    # 2) Clean up old MLflow‐model folder if it exists
    if os.path.exists(mlflow_model_dir):
        shutil.rmtree(mlflow_model_dir)

    # 3) Dump a simple pickle
    joblib.dump(rf, pickle_path)

    # 4) Infer signature & example
    input_example = pd.DataFrame(X_train[:1], columns=wine.feature_names)
    signature    = infer_signature(X_train, rf.predict(X_train))

    # 5) Save a full MLflow model locally
    mlflow.sklearn.save_model(
        sk_model=rf,
        path=mlflow_model_dir,
        signature=signature,
        input_example=input_example
    )

    # 6) Log both as artifacts under a single folder
    artifact_root = "Random-Forest-Model"
    mlflow.log_artifact(pickle_path,           artifact_path=artifact_root)
    mlflow.log_artifacts(mlflow_model_dir,     artifact_path=artifact_root)

    print(f"Artifacts logged under '{artifact_root}' for run {run_id}")

    print(accuracy)