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
    def get_data(self) -> Any:
        """Fetch the complete dataset (before any splitting)."""
        pass
