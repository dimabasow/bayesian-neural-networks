from typing import Any, Dict, List, Optional

import polars as pl

from src.data.preprocessing.metadata import Metadata
from src.data.preprocessing.transformers.base import BaseTransformer


class MissingIndicator(BaseTransformer):
    def __init__(
        self,
        columns: Optional[List[str]] = None,
    ):
        self.columns_in = list(columns)

    @property
    def columns_out(self) -> List[str]:
        return [f"missing_indicator_{column}" for column in self.columns_in]

    @classmethod
    def from_config(
        cls,
        columns: Optional[List[str]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> "MissingIndicator":
        return cls(columns=columns)

    @property
    def metadata(self) -> Dict[str, Any]:
        return Metadata(features_numeric=tuple(self.columns_out))

    @property
    def state(self) -> Dict[str, Any]:
        return {"columns": self.columns_in}

    def fit(self, data: pl.DataFrame):
        self.update_columns_in(data=data)

    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        list_to_cat = []
        for name_in, name_out in zip(self.columns_in, self.columns_out):
            series_encoded: pl.Series = (data[name_in].is_null().cast(pl.Int64)).rename(
                name_out
            )
            list_to_cat.append(series_encoded.to_frame())
        df_encoded = pl.concat(list_to_cat, how="horizontal")
        return df_encoded[self.columns_out]

    @staticmethod
    def filter_raw_data(data: pl.DataFrame) -> pl.DataFrame:
        columns_to_drop = []
        for column in data.columns:
            series = data[column]
            series_drop_null = series.drop_nulls().drop_nans()
            if (series_drop_null.len() == 0) or (
                series_drop_null.len() == series.len()
            ):
                columns_to_drop.append(column)
        data = data.drop(*columns_to_drop)
        return data
