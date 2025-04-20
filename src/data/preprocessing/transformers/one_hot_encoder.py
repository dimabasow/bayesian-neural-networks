import copy
from typing import Any, Dict, List, Optional

import polars as pl

from src.data.preprocessing.metadata import TransformType
from src.data.preprocessing.transformers.base import BaseTransformer
from src.data.preprocessing.utils import drop_columns_empty_or_constant


class OneHotEncoder(BaseTransformer):
    transform_type = TransformType.features_numeric

    def __init__(
        self,
        conf: List[Dict[str, Any]],
        min_frequency: Optional[int] = None,
        max_categories: Optional[int] = None,
    ):
        self.conf = copy.deepcopy(conf)
        self.columns_in = [item["column"] for item in conf]
        self.min_frequency = min_frequency
        self.max_categories = max_categories

    @property
    def columns_out(self) -> List[str]:
        return [
            self.rename_column(column=item["column"], value=item["value"])
            for item in self.conf
            if "value" in item
        ]

    @classmethod
    def from_config(
        cls,
        columns: Optional[List[str]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> "OneHotEncoder":
        if kwargs is None:
            kwargs = {}
        return cls(conf=[{"column": column} for column in columns], **kwargs)

    @property
    def state(self) -> Dict[str, Any]:
        return {
            "conf": self.conf,
            "min_frequency": self.min_frequency,
            "max_categories": self.max_categories,
        }

    def fit(self, data: pl.DataFrame):
        self.update_columns_in(data=data)
        conf = []
        for column in self.columns_in:
            df_count = data[column].drop_nans().drop_nulls().value_counts()
            if self.min_frequency is not None:
                df_count = df_count.filter(pl.col("count") >= self.min_frequency)
            if self.max_categories:
                df_count = df_count[: self.max_categories]
            values = df_count[column].to_list()
            for value in values:
                item = {"column": column, "value": value}
                conf.append(item)
        self.conf = conf

    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        list_to_cat = []
        for item in self.conf:
            column = item["column"]
            value = item["value"]
            series_encoded: pl.Series = (data[column] == value).cast(pl.Int64)
            series_encoded = series_encoded.rename(
                self.rename_column(column=column, value=value)
            )
            list_to_cat.append(series_encoded.to_frame())

        df_encoded: pl.DataFrame = pl.concat(list_to_cat, how="horizontal")
        df_encoded = df_encoded.fill_nan(0).fill_null(0)
        return df_encoded[self.columns_out]

    @staticmethod
    def filter_raw_data(data: pl.DataFrame) -> pl.DataFrame:
        return drop_columns_empty_or_constant(df=data)

    def rename_column(self, column: str, value) -> str:
        return f"{self.__class__.__name__}_{column}_{value}"
