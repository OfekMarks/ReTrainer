from typing import Any, List
from preprocessors.preprocessor_interface import PreprocessorInterface


class PreprocessingPipeline(PreprocessorInterface):
    """
    Composes an ordered list of preprocessing stages into a single pipeline.
    Each stage's output is fed as input to the next.
    """

    def __init__(self, stages: List[PreprocessorInterface]):
        self.stages = stages

    def preprocess(self, data: Any) -> Any:
        for stage in self.stages:
            data = stage.preprocess(data)
        return data
