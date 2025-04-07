import copy
from typing import Any, Dict, List, NamedTuple, Optional

import polars as pl
import scipy
import scipy.stats

from src.data.preprocessing.metadata import Metadata
from src.data.preprocessing.transformers.base import BaseTransformer


class YeoJohsonResult(NamedTuple):
    series: pl.Series
    lmbda: float
    mean: float
    std: float


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


class PowerTransformer(BaseTransformer):
    def __init__(
        self,
        conf: List[Dict[str, Any]],
    ):
        self.conf = conf
        self.columns_in = [item["column"] for item in conf]

    @property
    def columns_out(self) -> List[str]:
        return [item["name"] for item in self.conf if "name" in item]

    @classmethod
    def from_config(
        cls,
        columns: Optional[List[str]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> "PowerTransformer":
        return cls(conf=[{"column": column} for column in columns])

    @property
    def metadata(self) -> Metadata:
        return Metadata(features_numeric=tuple(self.columns_out))

    @property
    def state(self) -> Dict[str, Any]:
        return {"conf": copy.deepcopy(self.conf)}

    def fit_transform(self, data: pl.DataFrame) -> pl.DataFrame:
        self.update_columns_in(data=data)
        df = data[list(self.columns_in)].cast(pl.Float64)

        conf = []
        for column in df.columns:
            result = apply_yeojohson(x=df[column])
            df = df.with_columns(result.series)
            item = {
                "column": column,
                "name": f"{column}_PowerTransformer",
                "lmbda": result.lmbda,
                "mean": result.mean,
                "std": result.std,
            }
            conf.append(item)
        self.conf = conf

        df.columns = [f"{column}_PowerTransformer" for column in df.columns]
        df = df.fill_nan(0).fill_null(0)
        return df[self.columns_out]

    def fit(self, data: pl.DataFrame):
        self.fit_transform(data=data)

    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        df: pl.DataFrame = data[self.columns_in].cast(pl.Float64)
        for item in self.conf:
            df = df.with_columns(
                apply_yeojohson(
                    x=df[item["column"]],
                    lmbda=item["lmbda"],
                    mean=item["mean"],
                    std=item["std"],
                ).series
            )
        df.columns = [f"{column}_PowerTransformer" for column in df.columns]
        df = df.fill_nan(0).fill_null(0)
        return df[self.columns_out]

    @staticmethod
    def filter_raw_data(data: pl.DataFrame) -> pl.DataFrame:
        columns_to_drop = []
        for column in data.columns:
            series = data[column]
            series_drop_null = series.drop_nans().drop_nulls()
            if series_drop_null.len() == 0:
                columns_to_drop.append(column)
            elif (series_drop_null[0] == series_drop_null).all():
                columns_to_drop.append(column)
        data = data.drop(*columns_to_drop)
        return data
