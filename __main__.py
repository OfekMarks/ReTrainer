from trainer import run_training_pipeline
from trackers.mlflow_tracker import MLflowTracker
from data_loaders.dagshub_loader import DagsHubDataEngineLoader
from preprocessors.default_tabular_preprocessor import DefaultTabularPreprocessor

REPO = "ofekmarks/my-first-repo"
DATASOURCE = "datasource"


if __name__ == "__main__":
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
        preprocessor=DefaultTabularPreprocessor(),
    )
