import copy
from typing import Any, Dict, List, Optional

import polars as pl

from src.data.preprocessing.metadata import TransformType
from src.data.preprocessing.transformers.base import BaseTransformer
from src.data.preprocessing.utils import drop_columns_constant


class BinaryEncoder(BaseTransformer):
    transform_type = TransformType.features_numeric

    def __init__(self, conf: List[Dict[str, Any]]):
        self.conf = copy.deepcopy(conf)
        self.columns_in = [item["column"] for item in conf]

    @property
    def columns_out(self) -> List[str]:
        return [
            self.rename_column(
                column=item["column"],
                value_neg=item["value_neg"],
                value_pos=item["value_pos"],
            )
            for item in self.conf
            if "value_pos" in item and "value_neg" in item
        ]

    @classmethod
    def from_config(
        cls,
        columns: Optional[List[str]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> "BinaryEncoder":
        return cls(conf=[{"column": column} for column in columns])

    @property
    def state(self) -> Dict[str, Any]:
        return {"conf": copy.deepcopy(self.conf)}

    def fit(self, data: pl.DataFrame):
        self.update_columns_in(data=data)
        conf = []
        for column in self.columns_in:
            series = data[column].drop_nulls().drop_nans()
            value_neg, value_pos, *_ = series.value_counts(sort=True)[column]
            if {value_pos, value_neg} == {0, 1}:
                value_neg = 0
                value_pos = 1
            item = {
                "column": column,
                "value_neg": value_neg,
                "value_pos": value_pos,
            }
            conf.append(item)
        self.conf = conf

    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        list_to_cat: List[pl.DataFrame] = []
        for item in self.conf:
            column = item["column"]
            value_neg = item["value_neg"]
            value_pos = item["value_pos"]
            name = self.rename_column(
                column=column,
                value_neg=value_neg,
                value_pos=value_pos,
            )

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
        return drop_columns_constant(df=data)

    def rename_column(self, column: str, value_neg, value_pos) -> str:
        return f"{column}_{self.__class__.__name__}_{value_neg}_{value_pos}"
