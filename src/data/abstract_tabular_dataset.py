import random
from abc import ABC, abstractmethod
from typing import Dict, Iterator, List, NamedTuple, Optional, Sequence, Tuple, TypeVar

import polars as pl

from src.data.preprocessing.metadata import Metadata

TableVar = TypeVar("TableVar")


class TargetItem(NamedTuple):
    value: TableVar
    mask: Optional[TableVar] = None


class TableItem(NamedTuple):
    index: Optional[pl.DataFrame] = None
    features_numeric: Optional[TableVar] = None
    target: Optional[Dict[str, TargetItem]] = None


class PolarsTargetItem(TargetItem):
    value: pl.Series
    mask: Optional[pl.Series]


class PolarsTableItem(TableItem):
    index: Optional[pl.DataFrame]
    features_numeric: Optional[pl.DataFrame]
    features_category: Optional[Dict[str, pl.Series]]
    target: Optional[Dict[str, PolarsTargetItem]]


def filter_df_metadata(
    df: pl.DataFrame,
    metadata: Metadata,
) -> Tuple[pl.DataFrame, Metadata]:
    dict_metadata = metadata._asdict()
    columns_metadata = []
    for key in dict_metadata:
        key_columns = dict_metadata[key]
        if key_columns is not None:
            dict_metadata[key] = tuple(
                column for column in key_columns if column in df.columns
            )
            columns_metadata.extend(dict_metadata[key])
    columns = [column for column in df.columns if column in columns_metadata]
    df = df[columns]

    return df, metadata


def transform_df_to_table_item(df: pl.DataFrame, metadata: Metadata) -> PolarsTableItem:
    if metadata.index is None:
        df_index = None
    else:
        df_index = df[list(metadata.index)]

    if metadata.features_numeric is None:
        df_features = None
    else:
        df_features = df[list(metadata.features_numeric)]

    columns_target = []
    if metadata.targets_binary is not None:
        columns_target.extend(metadata.targets_binary)
    if metadata.targets_regression is not None:
        columns_target.extend(metadata.targets_regression)
    if metadata.targets_multiclass is not None:
        columns_target.extend(metadata.targets_multiclass)

    dict_target = {}
    for column in columns_target:
        series: pl.Series = df[column]
        if series.has_nulls():
            mask = series.is_null()
        else:
            mask = None
        dict_target[column] = PolarsTargetItem(value=series, mask=mask)
    if not dict_target:
        dict_target = None

    data = PolarsTableItem(
        index=df_index,
        features_numeric=df_features,
        target=dict_target,
    )
    return data


class AbstractTabularDataset(ABC):
    length: int
    metadata: Metadata
    data: TableItem
    target_dim: Dict[str, int]

    @abstractmethod
    def prepare_data(self, data: PolarsTableItem) -> TableItem:
        pass

    def __init__(
        self,
        df: pl.DataFrame,
        metadata: Metadata,
    ):
        df, metadata = filter_df_metadata(df, metadata)
        self.metadata = metadata
        self.length = len(df)
        data = transform_df_to_table_item(df=df, metadata=metadata)
        self.data = self.prepare_data(data=data)

    @property
    def target_dim(self, df: pl.DataFrame) -> Dict[str, int]:
        target_dim = {}
        for name in self.metadata.targets_binary:
            target_dim[name] = 1
        for name in self.metadata.targets_regression:
            target_dim[name] = 1
        for name in self.metadata.targets_multiclass:
            target_dim[name] = df[name].max() + 1

        return target_dim

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx) -> TableItem:
        if isinstance(idx, (slice, list)):
            idx = idx
        elif isinstance(idx, Sequence):
            idx = list(idx)
        else:
            idx = [idx]

        if self.data.index is None:
            index = None
        else:
            index = self.data.index[idx]

        if self.data.features_numeric is None:
            features_numeric = None
        else:
            features_numeric = self.data.features_numeric[idx]

        if self.data.target is None:
            target = None
        else:
            target = {}
            for name in self.data.target:
                item = self.data.target[name]
                value = item.value[idx]
                if item.mask is None:
                    mask = None
                else:
                    mask = item.mask[idx]
                target[name] = TargetItem(value=value, mask=mask)

        return TableItem(
            index=index,
            features_numeric=features_numeric,
            target=target,
        )

    def to_epochs(
        self,
        batch_size: int,
        shuffle: bool = False,
        num_epochs: Optional[int] = 1,
    ) -> Iterator[Iterator[TableItem]]:
        idx = list(range(len(self)))
        count = 0
        while num_epochs is None or count < num_epochs:
            count += 1
            if shuffle:
                random.shuffle(idx)
            yield self.__get_batches(idx=idx, batch_size=batch_size)

    def to_bathes(
        self,
        batch_size: int,
        shuffle: bool = False,
        num_epochs: Optional[int] = 1,
    ) -> Iterator[TableItem]:
        idx = list(range(len(self)))
        count = 0
        while num_epochs is None or count < num_epochs:
            count += 1
            if shuffle:
                random.shuffle(idx)
            yield from self.__get_batches(idx=idx, batch_size=batch_size)

    def __get_batches(
        self,
        idx: List[int],
        batch_size: int,
    ) -> Iterator[TableItem]:
        for i in range(0, len(idx), batch_size):
            idx_batch = idx[i : i + batch_size]
            yield self[idx_batch]
