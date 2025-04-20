from typing import Any, Dict, List, Optional

import polars as pl
import polars_hash as plh

from src.data.preprocessing.metadata import TransformType
from src.data.preprocessing.transformers.base import BaseTransformer
from src.data.preprocessing.utils import drop_columns_empty_or_constant


class HashEncoder(BaseTransformer):
    transform_type = TransformType.features_numeric

    def __init__(
        self,
        columns: Optional[List[str]],
        n_columns_encoded: int = 32,
    ):
        self.columns_in = list(columns)
        if n_columns_encoded > 32:
            raise ValueError(
                f"Max value for n_columns_encoded is 32, but {n_columns_encoded} was set"
            )
        self.n_columns_encoded = n_columns_encoded

    @property
    def columns_out(self) -> List[str]:
        columns_out = []
        for column in self.columns_in:
            for i in range(self.n_columns_encoded):
                columns_out.append(self.rename_column(column=column, count=i))
        return columns_out

    @classmethod
    def from_config(
        cls,
        columns: Optional[List[str]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> "HashEncoder":
        if kwargs is None:
            kwargs = {}
        return cls(columns=columns, **kwargs)

    @property
    def state(self) -> Dict[str, Any]:
        return {
            "columns": self.columns_in,
            "n_columns_encoded": self.n_columns_encoded,
        }

    def fit(self, data: pl.DataFrame):
        self.update_columns_in(data=data)

    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        df_encoded = data.select(
            *[
                plh.col(column)
                .cast(pl.String())
                .fill_null("<fcjdLR")
                .nchash.murmur32(seed=0)
                for column in self.columns_in
            ]
        )
        for column in self.columns_in:
            for count in range(self.n_columns_encoded):
                column_encoded = self.rename_column(column=column, count=count)
                df_encoded = df_encoded.with_columns(
                    (pl.col(column) % 2).alias(column_encoded)
                )
                df_encoded = df_encoded.with_columns(pl.col(column) // 2)
        return df_encoded[self.columns_out]

    @staticmethod
    def filter_raw_data(data: pl.DataFrame) -> pl.DataFrame:
        return drop_columns_empty_or_constant(df=data)

    def rename_column(self, column: str, count: int) -> str:
        return f"{self.__class__.__name__}_{column}_{count}"
