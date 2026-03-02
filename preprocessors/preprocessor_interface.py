from abc import ABC, abstractmethod
from typing import Any, Tuple


class PreprocessorInterface(ABC):
    """
    Abstract interface for data preprocessing and feature engineering.
    Adheres to the Open/Closed Principle: Users can pass any preprocessor
    implementation into the trainer without modifying the core pipeline script.
    """

    @abstractmethod
    def preprocess(
        self, train_data: Any, test_data: Any, target_column: str
    ) -> Tuple[Any, Any, Any, Any]:
        """
        Preprocess the train and test data.

        Args:
            train_data: The raw training data from the DataLoader.
            test_data: The raw testing data from the DataLoader.
            target_column: The name of the target column/feature.

        Returns:
            Tuple containing: (X_train, X_test, y_train, y_test)
        """
        pass
