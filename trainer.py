import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from trackers.tracker_interface import ExperimentTracker
from evaluator import Evaluator
from data_loaders.loader_interface import DataLoaderInterface
from preprocessors.preprocessor_interface import PreprocessorInterface


def run_training_pipeline(
    repo_name: str,
    datasource_name: str,
    metrics_to_log: list,
    target_column: str,
    tracker: ExperimentTracker,
    data_loader: DataLoaderInterface,
    preprocessor: PreprocessorInterface,
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

    # 3. Preprocess Data and Engineer Features
    print(f"Preprocessing data via '{preprocessor.__class__.__name__}'...")
    X_train, X_test, y_train, y_test = preprocessor.preprocess(
        train_data, test_data, target_column
    )

    print(
        f"Processed train split: {X_train.shape[0]} samples. Test split: {X_test.shape[0]} samples. Target: '{target_column}'"
    )

    # 4. Train the Model
    print("Training Random Forest Classifier on CPU...")
    # These are hyperparams we will log to mlflow
    model_params = {"n_estimators": 100, "max_depth": 10, "random_state": 42}
    tracker.log_params(
        dataset_repo=repo_name, datasource_name=datasource_name, **model_params
    )

    model = RandomForestClassifier(**model_params)
    model.fit(X_train, y_train)

    # 5. Evaluate and Log to Tracker
    print(f"Evaluating and logging to '{tracker.__class__.__name__}'...")

    y_pred_classes = model.predict(X_test)
    y_pred_proba = (
        model.predict_proba(X_test)[:, 1]
        if len(model.classes_) == 2
        else model.predict_proba(X_test)
    )

    evaluator = Evaluator(tracker=tracker)

    metrics_proba = [m for m in metrics_to_log if "ROC" in m]
    metrics_classes = [m for m in metrics_to_log if m not in metrics_proba]

    if metrics_proba:
        evaluator.log_metrics(metrics_proba, y_test, y_pred_proba, step=1)
    if metrics_classes:
        evaluator.log_metrics(metrics_classes, y_test, y_pred_classes, step=1)

    tracker.log_model(model, name="model", flavor="sklearn")
    print("Model logged successfully!")

    tracker.end_run()
    print("--- Training Pipeline Complete ---")
