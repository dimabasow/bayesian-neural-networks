import copy
from typing import Any, Dict, List, Optional

import polars as pl

from src.data.preprocessing.metadata import TransformType
from src.data.preprocessing.transformers.base import BaseTransformer
from src.data.preprocessing.utils import drop_columns_empty_or_constant


class BinaryTarget(BaseTransformer):
    transform_type = TransformType.targets_binary

    def __init__(self, conf: List[Dict[str, Any]]):
        self.conf = copy.deepcopy(conf)
        self.columns_in = [item["column"] for item in conf]

    @property
    def columns_out(self) -> List[str]:
        return [
            self.rename_column(
                column=item["column"],
                value_zero=item["value_zero"],
                value_one=item["value_one"],
            )
            for item in self.conf
            if "value_zero" in item and "value_one" in item
        ]

    @classmethod
    def from_config(
        cls,
        columns: Optional[List[str]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> "BinaryTarget":
        return cls(conf=[{"column": column} for column in columns])

    @property
    def state(self) -> Dict[str, Any]:
        return {"conf": copy.deepcopy(self.conf)}

    def fit(self, data: pl.DataFrame):
        self.update_columns_in(data=data)
        conf = []
        for column in self.columns_in:
            series = data[column].drop_nulls().drop_nans()
            value_zero, value_one, *_ = series.value_counts(sort=True)[column]
            if {value_zero, value_one} == {0, 1}:
                value_zero = 0
                value_one = 1
            item = {
                "column": column,
                "value_zero": value_zero,
                "value_one": value_one,
            }
            conf.append(item)
        self.conf = conf

    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        list_to_cat: List[pl.DataFrame] = []
        for item in self.conf:
            column = item["column"]
            if column in data.columns:
                value_zero = item["value_zero"]
                value_one = item["value_one"]
                name = self.rename_column(
                    column=column,
                    value_zero=value_zero,
                    value_one=value_one,
                )

                series_encoded = data.with_columns(
                    (
                        pl.when(pl.col(column) == value_zero)
                        .then(0)
                        .when(pl.col(column) == value_one)
                        .then(1)
                        .cast(pl.Int64())
                        .alias(name)
                    )
                )[name]
                list_to_cat.append(series_encoded.to_frame())
        df_encoded = pl.concat(list_to_cat, how="horizontal")
        return df_encoded

    @staticmethod
    def filter_raw_data(data: pl.DataFrame) -> pl.DataFrame:
        return drop_columns_empty_or_constant(df=data)

    def rename_column(self, column: str, value_zero, value_one) -> str:
        return f"{column}_{self.__class__.__name__}_{value_zero}_{value_one}"
