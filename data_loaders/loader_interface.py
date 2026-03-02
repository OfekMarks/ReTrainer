from abc import ABC, abstractmethod
from typing import Any


class DataLoaderInterface(ABC):
    """
    Abstract interface for loading datasets.
    Adheres to the Dependency Inversion Principle, decoupling the core
    training pipeline from specific data storage backends (DagsHub, S3, Local)
    and specific data structures (PyTorch, TensorFlow, Pandas).
    """

    @abstractmethod
    def get_train_data(self) -> Any:
        """Fetch the training dataset split."""
        pass

    @abstractmethod
    def get_test_data(self) -> Any:
        """Fetch the testing dataset split."""
        pass
