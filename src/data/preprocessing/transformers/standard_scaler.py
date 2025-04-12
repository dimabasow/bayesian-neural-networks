from abc import ABC
from typing import Any, Dict, List, NamedTuple, Optional

import polars as pl

from src.data.preprocessing.metadata import TransformType
from src.data.preprocessing.transformers.base import BaseTransformer
from src.data.preprocessing.utils import drop_columns_empty_or_constant


class FitTransformResult(NamedTuple):
    df: pl.DataFrame
    conf: List[Dict[str, Any]]


def transform(df: pl.DataFrame, conf: List[Dict[str, Any]]) -> pl.DataFrame:
    df = df.cast(pl.Float64())
    columns = []
    list_to_cat = []
    for item in conf:
        column = item["column"]
        if column in df.columns:
            columns.append(column)
            series = df[column]
            series = (series - item["mean"]) / item["std"]
            list_to_cat.append(series)
        df_ecoded: pl.DataFrame = pl.concat(list_to_cat, how="horizontal")
        return df_ecoded


class StandardScaler(BaseTransformer, ABC):
    def __init__(
        self,
        conf: List[Dict[str, Any]],
    ):
        self.conf = conf
        self.columns_in = [item["column"] for item in conf]

    @property
    def columns_out(self) -> List[str]:
        return [self.rename_column(item["name"]) for item in self.conf]

    @classmethod
    def from_config(
        cls,
        columns: Optional[List[str]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> "StandardScaler":
        return cls(conf=[{"column": column} for column in columns])

    @property
    def state(self) -> Dict[str, Any]:
        return {"conf": self.conf}

    def fit(self, data: pl.DataFrame):
        self.update_columns_in(data=data)
        data = data[self.columns_in].cast(pl.Float64)
        conf = []
        for column in data.columns:
            item = {
                "column": column,
                "mean": data[column].mean(),
                "std": data[column].std(),
            }
            conf.append(item)
        self.conf = conf

    @staticmethod
    def filter_raw_data(data: pl.DataFrame) -> pl.DataFrame:
        return drop_columns_empty_or_constant(df=data)

    def rename_column(self, column: str) -> str:
        return f"{column}_{self.__class__.__name__}"


class FeatureStandardScaler(StandardScaler):
    transform_type = TransformType.features_numeric

    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        df = transform(df=data[self.columns_in], conf=self.conf)
        df = df.fill_nan(0).fill_null(0)
        return df[self.columns_out]


class TargetStandardScaler(StandardScaler):
    transform_type = TransformType.targets_regression

    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        df = transform(df=data[self.columns_in], conf=self.conf)
        return df
