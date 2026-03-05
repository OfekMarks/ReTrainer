import pandas as pd
from typing import Any
from pydantic import BaseModel, Field
from preprocessors.preprocessor_interface import PreprocessorInterface


class DefaultTabularPreprocessor(PreprocessorInterface):
    """
    Concrete preprocessor for standard tabular DataFrames.
    Encodes categorical string columns using one-hot encoding (pd.get_dummies).
    """

    class ConfigModel(BaseModel):
        pass

    def __init__(self):
        pass

    def preprocess(self, data: Any) -> Any:
        return pd.get_dummies(data)
