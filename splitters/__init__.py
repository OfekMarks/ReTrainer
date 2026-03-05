from .random_splitter import RandomSplitter
from .column_splitter import ColumnSplitter
from .target_splitter import TargetSplitter

AVAILABLE_SPLITTERS = {
    "Random Split": RandomSplitter,
    "Column-Based Split": ColumnSplitter,
}
