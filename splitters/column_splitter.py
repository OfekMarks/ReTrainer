from typing import Any, Tuple
from pydantic import BaseModel, Field
from splitters.splitter_interface import DataSplitter


class ColumnSplitter(DataSplitter):
    """
    Splits data based on a column's value.
    Rows where the column matches `test_value` go to the test set; the rest go to train.
    The split column is dropped from both sets after splitting.
    """

    class ConfigModel(BaseModel):
        split_column: str = Field(
            default="split", description="Column name to split on"
        )
        test_value: str = Field(
            default="test", description="Value in the split column that indicates test data"
        )

    def __init__(self, split_column: str = "split", test_value: str = "test"):
        self.split_column = split_column
        self.test_value = test_value

    def split(self, data: Any) -> Tuple[Any, Any]:
        test_data = data[data[self.split_column] == self.test_value].drop(
            columns=[self.split_column]
        )
        train_data = data[data[self.split_column] != self.test_value].drop(
            columns=[self.split_column]
        )
        return train_data, test_data
