from abc import ABC
from typing import Any, Dict, List, Optional

import polars as pl

from src.data.preprocessing.metadata import TransformType
from src.data.preprocessing.transformers.base import BaseTransformer
from src.data.preprocessing.utils import drop_columns_empty_or_constant


class Numeric(BaseTransformer, ABC):
    def __init__(
        self,
        conf: List[Dict[str, Any]],
    ):
        self.conf = conf
        self.columns_in = [item["column"] for item in conf]

    @property
    def columns_out(self) -> List[str]:
        return [self.rename_column(column=item["column"]) for item in self.conf]

    @classmethod
    def from_config(
        cls,
        columns: Optional[List[str]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> "Numeric":
        return cls(conf=[{"column": column} for column in columns])

    @property
    def state(self) -> Dict[str, Any]:
        return {"conf": self.conf}

    def fit(self, data: pl.DataFrame):
        self.update_columns_in(data=data)

    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        df = data[self.columns_in].cast(pl.Float64)
        df.columns = [self.rename_column(column=column) for column in df.columns]
        return df[self.columns_out]

    @staticmethod
    def filter_raw_data(data: pl.DataFrame) -> pl.DataFrame:
        return drop_columns_empty_or_constant(df=data)

    def rename_column(self, column: str) -> str:
        return f"{self.__class__.__name__}_{column}"


class NumericFeature(Numeric):
    transform_type = TransformType.features_numeric


class NumericTarget(Numeric):
    transform_type = TransformType.targets_regression
