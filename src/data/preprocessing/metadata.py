from typing import NamedTuple, Tuple, Optional


class Metadata(NamedTuple):
    ids: Optional[Tuple[str, ...]] = None
    features_numeric: Optional[Tuple[str, ...]] = None
    features_category: Optional[Tuple[str, ...]] = None
    targets_regression: Optional[Tuple[str, ...]] = None
    targets_binary: Optional[Tuple[str, ...]] = None
    targets_multiclass: Optional[Tuple[str, ...]] = None
