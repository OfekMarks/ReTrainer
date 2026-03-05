from typing import Any, Tuple
from pydantic import BaseModel, Field
from sklearn.model_selection import train_test_split
from splitters.splitter_interface import DataSplitter


class RandomSplitter(DataSplitter):
    """
    Splits data randomly using sklearn's train_test_split.
    """

    class ConfigModel(BaseModel):
        test_size: float = Field(
            default=0.2, description="Fraction of data to use for testing (0.0–1.0)"
        )
        random_state: int = Field(
            default=42, description="Random seed for reproducibility"
        )

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state

    def split(self, data: Any) -> Tuple[Any, Any]:
        train_data, test_data = train_test_split(
            data, test_size=self.test_size, random_state=self.random_state
        )
        return train_data, test_data
