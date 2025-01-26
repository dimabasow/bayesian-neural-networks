from typing import NamedTuple, Dict, Literal, Optional, TypeVar
import polars as pl
import torch


class MetaDataColumn(NamedTuple):
    type: Literal["identifier", "numeric", "binary", "category"]
    target: bool = False
    feature: bool = False
    category_embeding_dim: Optional[int] = None
    target_masked_by: Optional[str] = None


TableVar = TypeVar("TableVar")
ColumnVar = TypeVar("ColumnVar")
MetaData = Dict[str, MetaDataColumn]
EmbedingInit = Dict[
    str,
    Dict[
        Literal["num_embeddings", "embedding_dim"],
        int
    ]
]


class TargetItem(NamedTuple):
    value: ColumnVar
    mask: Optional[ColumnVar] = None


class TableItem(NamedTuple):
    index: Optional[pl.DataFrame] = None
    features_numeric: Optional[TableVar] = None
    features_category: Optional[Dict[str, ColumnVar]] = None
    target: Optional[Dict[str, TargetItem]] = None


class PolarsTargetItem(TargetItem):
    value: pl.Series
    mask: Optional[pl.Series]


class PolarsTableItem(TableItem):
    index: Optional[pl.DataFrame]
    features_numeric: Optional[pl.DataFrame]
    features_category: Optional[Dict[str, pl.Series]]
    target: Optional[Dict[str, PolarsTargetItem]]


class TorchTargetItem(TargetItem):
    value: torch.Tensor
    mask: Optional[torch.BoolTensor]


class TorchTableItem(TableItem):
    index: Optional[pl.DataFrame]
    features_numeric: Optional[torch.Tensor]
    features_category: Optional[Dict[str, torch.LongTensor]]
    target: Optional[Dict[str, TorchTargetItem]]
