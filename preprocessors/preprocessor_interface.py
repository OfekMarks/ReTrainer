from abc import ABC, abstractmethod
from typing import Any


class PreprocessorInterface(ABC):
    """
    Abstract interface for a single data preprocessing stage.
    Each stage receives a single dataset and returns a transformed dataset.
    Stages are composable — multiple stages can be chained in a PreprocessingPipeline.
    """

    @abstractmethod
    def preprocess(self, data: Any) -> Any:
        """
        Apply a single preprocessing transformation.

        Args:
            data: The input dataset (e.g. a pandas DataFrame).

        Returns:
            The transformed dataset.
        """
        pass
