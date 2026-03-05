from trackers.tracker_interface import ExperimentTracker
from data_loaders.loader_interface import DataLoaderInterface
from preprocessors.preprocessing_pipeline import PreprocessingPipeline
from splitters.splitter_interface import DataSplitter
from splitters.target_splitter import TargetSplitter
from trainers.trainer_interface import TrainerInterface


def run_training_pipeline(
    tracker: ExperimentTracker,
    data_loader: DataLoaderInterface,
    preprocessing_pipeline: PreprocessingPipeline,
    splitter: DataSplitter,
    target_splitter: TargetSplitter,
    model_trainer: TrainerInterface,
):
    """
    Executes the full retraining pipeline:
    1. Loads the full dataset
    2. Runs ordered preprocessing stages
    3. Splits into train/test
    4. Extracts X/y
    5. Trains and evaluates
    """
    data = data_loader.get_data()

    data = preprocessing_pipeline.preprocess(data)

    train_data, test_data = splitter.split(data)

    X_train, X_test, y_train, y_test = target_splitter.split(train_data, test_data)

    model_trainer.train_and_evaluate(X_train, y_train, X_test, y_test)
