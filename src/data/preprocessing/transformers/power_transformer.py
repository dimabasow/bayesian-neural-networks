import copy
from abc import ABC
from typing import Any, Dict, List, NamedTuple, Optional

import polars as pl
import scipy
import scipy.stats

from src.data.preprocessing.metadata import TransformType
from src.data.preprocessing.transformers.base import BaseTransformer
from src.data.preprocessing.utils import drop_columns_empty_or_constant


class YeoJohsonResult(NamedTuple):
    series: pl.Series
    lmbda: float
    mean: float
    std: float


class FitTransformResult(NamedTuple):
    df: pl.DataFrame
    conf: List[Dict[str, Any]]


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
    lmbda: Optional[float] = None,
    mean: Optional[float] = None,
    std: Optional[float] = None,
) -> YeoJohsonResult:
    if lmbda is None:
        lmbda = scipy.stats.yeojohnson_normmax(
            x=x.drop_nulls().drop_nans().to_numpy()
        ).item()
    series = pl.DataFrame().with_columns(
        pl.when(x >= 0)
        .then(apply_yeojohson_pos(x=x, lmbda=lmbda))
        .otherwise(apply_yeojohson_neg(x=x, lmbda=lmbda))
        .alias(x.name)
    )[x.name]
    if mean is None:
        mean = series.mean()
    if std is None:
        std = series.std()
    series = (series - mean) / std
    return YeoJohsonResult(series=series, lmbda=lmbda, mean=mean, std=std)


def fit_transform(df: pl.DataFrame) -> FitTransformResult:
    df = df.cast(pl.Float64())
    conf = []
    for column in df.columns:
        result = apply_yeojohson(x=df[column])
        df = df.with_columns(result.series)
        item = {
            "column": column,
            "lmbda": result.lmbda,
            "mean": result.mean,
            "std": result.std,
        }
        conf.append(item)

    return FitTransformResult(df=df, conf=conf)


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
                    mean=item["mean"],
                    std=item["std"],
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
        self.fit_transform(data=data)

    @staticmethod
    def filter_raw_data(data: pl.DataFrame) -> pl.DataFrame:
        return drop_columns_empty_or_constant(df=data)

    def rename_column(self, column: str) -> str:
        return f"{column}_{self.__class__.__name__}"


class FeaturePowerTransformer(PowerTransformer):
    transform_type = TransformType.features_numeric

    def fit_transform(self, data: pl.DataFrame) -> pl.DataFrame:
        self.update_columns_in(data=data)
        result = fit_transform(df=data[list(self.columns_in)])
        self.conf = result.conf
        df = result.df

        df.columns = [self.rename_column(column=column) for column in df.columns]
        df = df.fill_nan(0).fill_null(0)
        return df[self.columns_out]

    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        df = transform(df=data[self.columns_in], conf=self.conf)
        df.columns = [self.rename_column(column=column) for column in df.columns]
        df = df.fill_nan(0).fill_null(0)
        return df[self.columns_out]


class TargetPowerTransformer(PowerTransformer):
    transform_type = TransformType.targets_regression

    def fit_transform(self, data: pl.DataFrame) -> pl.DataFrame:
        self.update_columns_in(data=data)
        result = fit_transform(df=data[list(self.columns_in)])
        self.conf = result.conf
        df = result.df

        df.columns = [self.rename_column(column=column) for column in df.columns]
        return df

    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        df = transform(df=data[self.columns_in], conf=self.conf)
        df.columns = [self.rename_column(column=column) for column in df.columns]
        return df
