from typing import Any, Dict, List, Optional

import polars as pl

from src.data.preprocessing.metadata import TransformType
from src.data.preprocessing.transformers.base import BaseTransformer


class Index(BaseTransformer):
    transform_type = TransformType.index

    def __init__(
        self,
        columns: Optional[List[str]] = None,
    ):
        self.columns_in = list(columns)

    @property
    def columns_out(self) -> List[str]:
        return list(self.columns_in)

    @classmethod
    def from_config(
        cls,
        columns: Optional[List[str]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> "Index":
        return cls(columns=columns)

    @property
    def state(self) -> Dict[str, Any]:
        return {"columns": self.columns_in}

    def fit(self, data: pl.DataFrame):
        self.update_columns_in(data=data)

    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        columns = []
        for column in data.columns:
            if column in self.columns_in:
                columns.append(column)
        return data[columns]

    @staticmethod
    def filter_raw_data(data: pl.DataFrame) -> pl.DataFrame:
        return data
