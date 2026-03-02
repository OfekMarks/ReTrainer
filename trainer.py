from trackers.tracker_interface import ExperimentTracker
from evaluator import Evaluator
from data_loaders.loader_interface import DataLoaderInterface
from preprocessors.preprocessor_interface import PreprocessorInterface
from trainers.trainer_interface import TrainerInterface


def run_training_pipeline(
    tracker: ExperimentTracker,
    data_loader: DataLoaderInterface,
    preprocessor: PreprocessorInterface,
    model_trainer: TrainerInterface,
):
    """
    Simulates a simplified model retraining pipeline.
    1. Loads dataset from Dagshub
    2. Trains a Scikit-Learn model on CPU
    3. Evaluates with the requested metrics
    4. Logs everything via the generic ExperimentTracker
    """
    train_data = data_loader.get_train_data()
    test_data = data_loader.get_test_data()

    X_train, X_test, y_train, y_test = preprocessor.preprocess(train_data, test_data)

    model_trainer.train_and_evaluate(X_train, y_train, X_test, y_test)
