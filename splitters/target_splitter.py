from typing import Any, Tuple
from pydantic import BaseModel, Field


class TargetSplitter:
    """
    Extracts features (X) and target (y) from train and test datasets.
    This is the final splitting step: (train, test) → (X_train, X_test, y_train, y_test).
    """

    class ConfigModel(BaseModel):
        target_column: str = Field(
            default="survived", description="The name of the column to predict"
        )

    def __init__(self, target_column: str = "survived"):
        self.target_column = target_column

    def split(self, train_data: Any, test_data: Any) -> Tuple[Any, Any, Any, Any]:
        """
        Extract features and target from train and test data.

        Returns:
            (X_train, X_test, y_train, y_test)
        """
        X_train = train_data.drop(columns=[self.target_column])
        y_train = train_data[self.target_column]
        X_test = test_data.drop(columns=[self.target_column])
        y_test = test_data[self.target_column]

        return X_train, X_test, y_train, y_test
