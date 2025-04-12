from enum import Enum
from typing import NamedTuple, Optional, Tuple


class Metadata(NamedTuple):
    ids: Optional[Tuple[str, ...]] = None
    features_numeric: Optional[Tuple[str, ...]] = None
    targets_regression: Optional[Tuple[str, ...]] = None
    targets_binary: Optional[Tuple[str, ...]] = None
    targets_multiclass: Optional[Tuple[str, ...]] = None


class TransformType(Enum):
    ids = 0
    features_numeric = 1
    targets_regression = 2
    targets_binary = 3
    targets_multiclass = 4
