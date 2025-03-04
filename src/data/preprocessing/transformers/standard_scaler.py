from typing import Dict, Any, List, Optional
import polars as pl
from src.data.preprocessing.transformers import BaseTransformer
from src.data.preprocessing import Metadata


class StandardScaler(BaseTransformer):
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
    ) -> "StandardScaler":
        return cls(conf=[{"column": column} for column in columns])

    @property
    def metadata(self) -> Metadata:
        return Metadata(features_numeric=tuple(self.columns_out))

    @property
    def state(self) -> Dict[str, Any]:
        return {
            "conf": self.conf
        }

    def fit(self, data: pl.DataFrame):
        self.update_columns_in(data=data)
        data = data[self.columns_in].cast(pl.Float64)
        conf = []
        for column in data.columns:
            item = {
                "column": column,
                "name": f"{column}_StandardScaler",
                "mean": data[column].mean(),
                "std": data[column].std(),
            }
            conf.append(item)
        self.conf = conf

    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        df: pl.DataFrame = data[self.columns_in].cast(pl.Float64)
        list_to_cat = []
        for item in self.conf:
            series = df[item["column"]]
            series = (series - item["mean"]) / item["std"]
            series = series.rename(item["name"])
            list_to_cat.append(series)
        df_ecoded: pl.DataFrame = pl.concat(list_to_cat, how="horizontal")
        df_ecoded = df_ecoded.fill_nan(0).fill_null(0)
        return df_ecoded[self.columns_out]

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
