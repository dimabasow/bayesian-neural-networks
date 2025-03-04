from typing import Dict, Any, List, Optional
import copy
import polars as pl
import scipy
import scipy.stats
from src.data.preprocessing.transformers import BaseTransformer
from src.data.preprocessing import Metadata


def apply_yeojohson_pos(x: pl.Series, lmbda: float) -> pl.Series:
    if lmbda != 0:
        return ((x + 1)**lmbda - 1) / lmbda
    else:
        return (x + 1).log()


def apply_yeojohson_neg(x: pl.Series, lmbda: float) -> pl.Series:
    if lmbda != 2:
        return -((-x + 1)**(2 - lmbda) - 1) / (2 - lmbda)
    else:
        return -(-x + 1).log()


class PowerTransformer(BaseTransformer):
    def __init__(
        self,
        conf: List[Dict[str, Any]],
    ):
        self.conf = conf
        self.columns_in = [item["column"] for item in conf]

    @property
    def columns_out(self) -> List[str]:
        return [
            item["name"]
            for item in self.conf
            if "name" in item
        ]

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
        return {
            "conf": copy.deepcopy(self.conf)
        }

    def fit_transform(self, data: pl.DataFrame) -> pl.DataFrame:
        self.update_columns_in(data=data)
        df = data[list(self.columns_in)].cast(pl.Float64)

        conf = []
        for column in df.columns:
            lmbda = scipy.stats.yeojohnson_normmax(
                x=df[column].drop_nulls().drop_nans().to_numpy()
            )
            df = df.with_columns(
                pl.when(pl.col(column) >= 0)
                .then(
                    apply_yeojohson_pos(x=pl.col(column), lmbda=lmbda)
                )
                .otherwise(
                    apply_yeojohson_neg(x=pl.col(column), lmbda=lmbda)
                )
                .alias(column)
            )
            mean = df[column].mean()
            std = df[column].std()
            df = df.with_columns(
                ((pl.col(column) - mean) / std).alias(column)
            )

            item = {
                "column": column,
                "name": f"{column}_PowerTransformer",
                "lmbda": lmbda,
                "mean": mean,
                "std": std,
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
            column = item["column"]
            lmbda = item["lmbda"]
            mean = item["mean"]
            std = item["std"]

            df = df.with_columns(
                pl.when(pl.col(column) >= 0)
                .then(
                    apply_yeojohson_pos(x=pl.col(column), lmbda=lmbda)
                )
                .otherwise(
                    apply_yeojohson_neg(x=pl.col(column), lmbda=lmbda)
                )
                .alias(column)
            )
            df = df.with_columns(
                ((pl.col(column) - mean) / std).alias(column)
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
