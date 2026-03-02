import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from settings import settings
import mlflow

# Integration Handlers
from trackers.tracker_interface import ExperimentTracker
from trackers.mlflow_tracker import MLflowTracker
from evaluator import Evaluator
from data_loaders.loader_interface import DataLoaderInterface
from data_loaders.dagshub_loader import DagsHubDataEngineLoader

# Authentication
import dagshub

dagshub.auth.add_app_token(settings.dagshub_user_token)


def run_training_pipeline(
    repo_name: str,
    datasource_name: str,
    metrics_to_log: list,
    target_column: str,
    tracker: ExperimentTracker,
    data_loader: DataLoaderInterface,
):
    """
    Simulates a simplified model retraining pipeline.
    1. Loads dataset from Dagshub
    2. Trains a Scikit-Learn model on CPU
    3. Evaluates with the requested metrics
    4. Logs everything via the generic ExperimentTracker
    """
    print(f"--- Starting Training Pipeline ---")

    # 1. Setup Experiment Tracking
    run_name = f"Retraining_{repo_name.split('/')[-1]}_{datasource_name}"
    tracker.setup_experiment(
        experiment_name="cpu_retraining_experiment", run_name=run_name
    )

    # 2. Load Data using Generic DataLoader Interface
    print(f"Loading data via '{data_loader.__class__.__name__}'...")
    train_data = data_loader.get_train_data()
    test_data = data_loader.get_test_data()

    X_train = train_data.drop(target_column, axis=1)
    y_train = train_data[target_column]
    X_test = test_data.drop(target_column, axis=1)
    y_test = test_data[target_column]

    # Standard string encoding if the target is categorical text
    if y_train.dtype == "object" or y_train.dtype.name == "category":
        y_train = y_train.astype("category").cat.codes
        y_test = y_test.astype("category").cat.codes

    # Standard numerical encoding for features if there are strings
    X_train = pd.get_dummies(X_train)
    X_test = pd.get_dummies(X_test)

    # Align columns in case some categories were only present in train or test
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    print(
        f"Loaded train split: {X_train.shape[0]} samples. Test split: {X_test.shape[0]} samples. Target: '{target_column}'"
    )

    # 3. Train the Model
    print("Training Random Forest Classifier on CPU...")
    # These are hyperparams we will log to mlflow
    params = {"n_estimators": 100, "max_depth": 10, "random_state": 42}

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    tracker.log_params(
        {
            "dataset_repo": repo_name,
            "datasource_name": datasource_name,
            **params,
        }
    )

    # 4. Evaluate and Log to Tracker
    print(f"Evaluating and logging to '{tracker.__class__.__name__}'...")

    evaluator = Evaluator(tracker=tracker)

    # Generate predictions
    # Note: 'ROC' requires probabilities, others might require hard classes
    # For simplicity of the dynamic metric system, we generate both and pass probability only for ROC
    y_pred_classes = model.predict(X_test)
    y_pred_proba = (
        model.predict_proba(X_test)[:, 1]
        if len(model.classes_) == 2
        else model.predict_proba(X_test)
    )

    # To use our dynamic system perfectly without complicated condition checking,
    # we can route the predictions based on what the metric expects.
    # In a real system, metric wrappers can specify if they need `y_pred` or `y_proba`.

    metrics_proba = [m for m in metrics_to_log if "ROC" in m]
    metrics_classes = [m for m in metrics_to_log if m not in metrics_proba]

    if metrics_proba:
        evaluator.log_metrics(metrics_proba, y_test, y_pred_proba, step=1)
    if metrics_classes:
        evaluator.log_metrics(metrics_classes, y_test, y_pred_classes, step=1)

    print(f"Evaluated {len(metrics_to_log)} metrics.")

    # Log the trained model
    tracker.log_model(model, name="model", flavor="sklearn")
    print("Model logged successfully!")

    tracker.end_run()
    print("--- Training Pipeline Complete ---")


if __name__ == "__main__":
    REPO = "ofekmarks/my-first-repo"
    DATASOURCE = "datasource"

    dagshub_loader = DagsHubDataEngineLoader(
        repo=REPO, datasource_name=DATASOURCE, format_type="dataframe"
    )

    run_training_pipeline(
        repo_name=REPO,
        datasource_name=DATASOURCE,
        metrics_to_log=["ROC", "Recall", "Precision"],
        target_column="survived",
        tracker=MLflowTracker(),
        data_loader=dagshub_loader,
    )
