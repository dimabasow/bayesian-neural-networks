import copy
from typing import Any, Dict, List, Optional

import polars as pl

from src.data.preprocessing.metadata import Metadata
from src.data.preprocessing.transformers.base import BaseTransformer


class BinaryEncoder(BaseTransformer):
    def __init__(self, conf: List[Dict[str, Any]]):
        self.conf = copy.deepcopy(conf)
        self.columns_in = [item["column"] for item in conf]

    @property
    def columns_out(self) -> List[str]:
        return [item["name"] for item in self.conf if "name" in item]

    @classmethod
    def from_config(
        cls,
        columns: Optional[List[str]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> "BinaryEncoder":
        return cls(conf=[{"column": column} for column in columns])

    @property
    def metadata(self) -> Metadata:
        return Metadata(features_numeric=tuple(self.columns_out))

    @property
    def state(self) -> Dict[str, Any]:
        return {"conf": copy.deepcopy(self.conf)}

    def fit(self, data: pl.DataFrame):
        self.update_columns_in(data=data)
        conf = []
        for column in self.columns_in:
            series = data[column].drop_nulls().drop_nans()
            value_pos, value_neg, *_ = series.value_counts(sort=True)[column]
            if {value_pos, value_neg} == {0, 1}:
                value_pos = 1
                value_neg = 0
            item = {
                "column": column,
                "name": f"binary_{column}_{value_neg}_{value_pos}",
                "value_pos": value_pos,
                "value_neg": value_neg,
            }
            conf.append(item)
        self.conf = conf

    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        list_to_cat: List[pl.DataFrame] = []
        for item in self.conf:
            column = item["column"]
            name = item["name"]
            value_pos = item["value_pos"]
            value_neg = item["value_neg"]

            series_encoded: pl.Series = (
                (data[column] == value_pos).cast(pl.Int64)
                - (data[column] == value_neg).cast(pl.Int64)
            ).rename(name)
            list_to_cat.append(series_encoded.to_frame())
        df_encoded = pl.concat(list_to_cat, how="horizontal")
        df_encoded = df_encoded.fill_nan(0).fill_null(0)
        return df_encoded[self.columns_out]

    @staticmethod
    def filter_raw_data(data: pl.DataFrame) -> pl.DataFrame:
        columns_to_drop = []
        for column in data.columns:
            series = data[column]
            series_drop_null = series.drop_nulls().drop_nans()
            if series_drop_null.len() == 0:
                columns_to_drop.append(column)
            elif (
                series_drop_null.len() == series.len()
                and (series_drop_null[0] == series_drop_null).all()
            ):
                columns_to_drop.append(column)
        data = data.drop(*columns_to_drop)
        return data
