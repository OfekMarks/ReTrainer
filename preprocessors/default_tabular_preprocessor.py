import pandas as pd
from typing import Any, Tuple
from preprocessors.preprocessor_interface import PreprocessorInterface


class DefaultTabularPreprocessor(PreprocessorInterface):
    """
    Concrete implementation of PreprocessorInterface for standard
    tabular DataFrames, specifically tailored for Scikit-Learn models.
    """

    def __init__(self, target_column: str):
        self.target_column = target_column

    def preprocess(self, train_data: Any, test_data: Any) -> Tuple[Any, Any, Any, Any]:
        X_train = train_data.drop(self.target_column, axis=1)
        y_train = train_data[self.target_column]
        X_test = test_data.drop(self.target_column, axis=1)
        y_test = test_data[self.target_column]

        # Standard string encoding if the target is categorical text
        if y_train.dtype == "object" or y_train.dtype.name == "category":
            y_train = y_train.astype("category").cat.codes
            y_test = y_test.astype("category").cat.codes

        # Standard numerical encoding for features if there are strings
        X_train = pd.get_dummies(X_train)
        X_test = pd.get_dummies(X_test)

        # Align columns in case some categories were only present in train or test
        X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

        return X_train, X_test, y_train, y_test
