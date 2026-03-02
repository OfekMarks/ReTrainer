from trainer import run_training_pipeline
from trackers.mlflow_tracker import MLflowTracker
from data_loaders.dagshub_loader import DagsHubDataEngineLoader
from preprocessors.default_tabular_preprocessor import DefaultTabularPreprocessor
from trainers.sklearn_classification_trainer import SklearnClassificationTrainer
from sklearn.ensemble import RandomForestClassifier

REPO = "ofekmarks/my-first-repo"
DATASOURCE = "datasource"


if __name__ == "__main__":
    dagshub_loader = DagsHubDataEngineLoader(
        repo=REPO, datasource_name=DATASOURCE, format_type="dataframe"
    )

    with MLflowTracker(
        experiment_name="cpu_retraining_experiment",
        run_name=f"retraining_{REPO.split('/')[-1]}_{DATASOURCE}",
    ) as tracker:

        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

        classification_trainer = SklearnClassificationTrainer(
            model=model,
            model_uri="models:/survival-classifier/1",
            tracker=tracker,
            metrics=["ROC", "Recall", "Precision"],
        )

        run_training_pipeline(
            tracker=tracker,
            data_loader=dagshub_loader,
            preprocessor=DefaultTabularPreprocessor(target_column="survived"),
            model_trainer=classification_trainer,
        )
