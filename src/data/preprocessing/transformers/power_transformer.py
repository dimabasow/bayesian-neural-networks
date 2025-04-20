import copy
from abc import ABC
from typing import Any, Dict, List, Optional

import polars as pl
import scipy
import scipy.stats

from src.data.preprocessing.metadata import TransformType
from src.data.preprocessing.transformers.base import BaseTransformer
from src.data.preprocessing.utils import drop_columns_empty_or_constant


def apply_yeojohson_pos(x: pl.Series, lmbda: float) -> pl.Series:
    if lmbda != 0:
        return ((x + 1) ** lmbda - 1) / lmbda
    else:
        return (x + 1).log()


def apply_yeojohson_neg(x: pl.Series, lmbda: float) -> pl.Series:
    if lmbda != 2:
        return -((-x + 1) ** (2 - lmbda) - 1) / (2 - lmbda)
    else:
        return -(-x + 1).log()


def apply_yeojohson(
    x: pl.Series,
    lmbda: float,
) -> pl.Series:
    series = pl.DataFrame().with_columns(
        pl.when(x >= 0)
        .then(apply_yeojohson_pos(x=x, lmbda=lmbda))
        .otherwise(apply_yeojohson_neg(x=x, lmbda=lmbda))
        .alias(x.name)
    )[x.name]
    return series


def transform(df: pl.DataFrame, conf: List[Dict[str, Any]]) -> pl.DataFrame:
    df = df.cast(pl.Float64())
    columns = []
    for item in conf:
        column = item["column"]
        if column in df.columns:
            columns.append(column)
            df = df.with_columns(
                apply_yeojohson(
                    x=df[column],
                    lmbda=item["lmbda"],
                ).series
            )
    df = df[columns]
    return df


class PowerTransformer(BaseTransformer, ABC):
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
    ) -> "PowerTransformer":
        return cls(conf=[{"column": column} for column in columns])

    @property
    def state(self) -> Dict[str, Any]:
        return {"conf": copy.deepcopy(self.conf)}

    def fit(self, data: pl.DataFrame):
        self.update_columns_in(data=data)
        data = data[self.columns_in].cast(pl.Float64)
        conf = []
        for column in data.columns:
            lmbda = scipy.stats.yeojohnson_normmax(
                x=data[column].drop_nulls().drop_nans().to_numpy()
            ).item()
            item = {
                "column": column,
                "lmbda": lmbda,
            }
            conf.append(item)
        self.conf = conf

    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        df = data[self.columns_in].cast(pl.Float64)
        for item in self.conf:
            column = item["column"]
            if column in df.columns:
                df = df.with_columns(
                    apply_yeojohson(
                        x=df[column],
                        lmbda=item["lmbda"],
                    )
                )
        df.columns = [self.rename_column(column=column) for column in df.columns]
        return df[self.columns_out]

    @staticmethod
    def filter_raw_data(data: pl.DataFrame) -> pl.DataFrame:
        return drop_columns_empty_or_constant(df=data)

    def rename_column(self, column: str) -> str:
        return f"{self.__class__.__name__}_{column}"


class PowerTransformerFeature(PowerTransformer):
    transform_type = TransformType.features_numeric


class PowerTransformerTarget(PowerTransformer):
    transform_type = TransformType.targets_regression
