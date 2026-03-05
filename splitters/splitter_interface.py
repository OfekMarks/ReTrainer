from abc import ABC, abstractmethod
from typing import Any, Tuple


class DataSplitter(ABC):
    """
    Abstract interface for splitting a dataset into train and test subsets.
    """

    @abstractmethod
    def split(self, data: Any) -> Tuple[Any, Any]:
        """
        Split the data into training and testing subsets.

        Args:
            data: The complete dataset.

        Returns:
            A tuple of (train_data, test_data).
        """
        pass
