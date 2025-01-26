from abc import ABC, abstractmethod
from typing import Dict, List, Iterator, Tuple, Sequence, Optional
import random
import polars as pl
from src.data.types import (
    MetaData,
    MetaDataColumn,
    TableItem,
    TargetItem,
    EmbedingInit,
    PolarsTableItem,
)


class AbstractTabularDataset(ABC):
    length: int
    metadata: MetaData
    data: TableItem
    target_dim: Dict[str, int]
    embeding_init_kwargs: EmbedingInit

    @abstractmethod
    def make_data(self, data: PolarsTableItem) -> TableItem:
        pass

    def __init__(
        self,
        df: pl.DataFrame,
        metadata: Dict[str, MetaDataColumn],
    ) -> pl.DataFrame:
        df, metadata = self.__filter_df_metadata(df, metadata)
        self.metadata = metadata

        df = self.__cast_df_types(df=df)
        self.target_dim = self.__get_target_dim(df=df)
        self.embeding_init_kwargs = self.__get_embeding_init_kwargs(df=df)
        self.length = len(df)
        data_init = self.__init_data(df=df)
        self.data = self.make_data(data=data_init)

    @staticmethod
    def __filter_df_metadata(
        df: pl.DataFrame,
        metadata: MetaData,
    ) -> Tuple[pl.DataFrame, MetaData]:
        metadata = {
            key: value
            for key, value in metadata.items()
            if key in df.columns
        }
        columns = sorted(metadata.keys())
        df = df[columns]
        return df, metadata

    def __cast_df_types(self, df: pl.DataFrame) -> pl.DataFrame:
        df = df.with_columns(
            pl.col(column).cast(pl.Boolean)
            for column in
            self.columns_binary
        )
        df = df.with_columns(
            pl.col(column).cast(pl.Float32)
            for column in
            self.columns_numeric
        )
        df = df.with_columns(
            pl.col(column).cast(pl.Int64)
            for column in
            self.columns_category
        )
        return df

    def __get_target_dim(self, df: pl.DataFrame) -> Dict[str, int]:
        target_dim = {}
        for name in self.columns_binary_target:
            target_dim[name] = 1
        for name in self.columns_numeric_target:
            target_dim[name] = 1
        for name in self.columns_category_target:
            target_dim[name] = df[name].max() + 1

        return target_dim

    def __get_embeding_init_kwargs(self, df: pl.DataFrame) -> EmbedingInit:
        embeding_init_kwargs = {}
        for name in self.columns_category_features:
            num_embeddings = df[name].max() + 1
            embedding_dim = self.metadata[name].category_embeding_dim
            embeding_init_kwargs[name] = {
                "num_embeddings": num_embeddings,
                "embedding_dim": embedding_dim,
            }

        return embeding_init_kwargs

    def __init_data(self, df: pl.DataFrame) -> PolarsTableItem:
        columns_numeric = (
            self.columns_numeric_features
            + self.columns_binary_features
        )

        index = df[self.columns_id]
        if index.is_empty():
            index = None

        features_numeric = df[columns_numeric].cast(pl.Float32)
        if features_numeric.is_empty():
            features_numeric = None

        features_category = {
            column: df[column]
            for column in self.columns_category_features
        }
        if not features_category:
            features_category = None

        target: Dict[str, TargetItem] = {}
        for column in self.columns_target:
            series = df[column]
            masked_by = self.metadata[column].target_masked_by
            if masked_by is None:
                mask = None
            else:
                mask = df[masked_by].cast(pl.Boolean)

            if series.has_nulls():
                if mask is None:
                    mask = series.is_null()
                else:
                    mask = mask * series.is_null()
            target[column] = TargetItem(value=series, mask=mask)
        if not target:
            target = None

        data = TableItem(
            index=index,
            features_numeric=features_numeric,
            features_category=features_category,
            target=target,
        )
        return data

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

        if self.data.features_category is None:
            features_category = None
        else:
            features_category = {
                column: self.data.features_category[column][idx]
                for column in self.data.features_category
            }

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
            features_category=features_category,
            target=target,
        )

    @staticmethod
    def transform_df_metadata(
        df: pl.DataFrame,
        metadata: Dict[str, MetaDataColumn]
    ) -> Tuple[pl.DataFrame, MetaData]:
        metadata = {
            key: value
            for key, value in metadata.items()
            if key in df.columns
        }
        columns = sorted(metadata.keys())
        df = df[columns]

        columns_binary = sorted(
            column
            for column in columns
            if metadata[column].type == "binary"
        )
        columns_numeric = sorted(
            column
            for column in columns
            if metadata[column].type == "numeric"
        )
        columns_category = sorted(
            column
            for column in columns
            if metadata[column].type == "category"
        )

        df = df[columns]
        df = df.with_columns(
            pl.col(column).cast(pl.Boolean)
            for column in
            columns_binary
        )
        df = df.with_columns(
            pl.col(column).cast(pl.Float32)
            for column in
            columns_numeric
        )
        df = df.with_columns(
            pl.col(column).cast(pl.Int64)
            for column in
            columns_category
        )

        return df, metadata

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
            idx_batch = idx[i: i + batch_size]
            yield self[*idx_batch]

    @property
    def columns(self) -> List[str]:
        return sorted(
            name for name in self.metadata
        )

    @property
    def columns_id(self) -> List[str]:
        return sorted(
            name for name in self.metadata
            if self.metadata[name].type == "identifier"
        )

    @property
    def columns_numeric(self) -> List[str]:
        return sorted(
            name for name in self.metadata
            if self.metadata[name].type == "numeric"
        )

    @property
    def columns_binary(self) -> List[str]:
        return sorted(
            name for name in self.metadata
            if self.metadata[name].type == "binary"
        )

    @property
    def columns_category(self) -> List[str]:
        return sorted(
            name for name in self.metadata
            if self.metadata[name].type == "category"
        )

    @property
    def columns_features(self) -> List[str]:
        return sorted(
            name for name in self.metadata
            if self.metadata[name].feature
        )

    @property
    def columns_target(self) -> List[str]:
        return sorted(
            name for name in self.metadata
            if self.metadata[name].target
        )

    @property
    def columns_numeric_features(self) -> List[str]:
        return sorted(
            set(self.columns_numeric) & set(self.columns_features)
        )

    @property
    def columns_numeric_target(self) -> List[str]:
        return sorted(
            set(self.columns_numeric) & set(self.columns_target)
        )

    @property
    def columns_binary_features(self) -> List[str]:
        return sorted(
            set(self.columns_binary) & set(self.columns_features)
        )

    @property
    def columns_binary_target(self) -> List[str]:
        return sorted(
            set(self.columns_binary) & set(self.columns_target)
        )

    @property
    def columns_category_features(self) -> List[str]:
        return sorted(
            set(self.columns_category) & set(self.columns_features)
        )

    @property
    def columns_category_target(self) -> List[str]:
        return sorted(
            set(self.columns_category) & set(self.columns_target)
        )
