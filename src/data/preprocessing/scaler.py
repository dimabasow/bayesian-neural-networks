from typing import Dict, Optional

import polars as pl

from src.data.preprocessing.metadata import Metadata
from src.data.preprocessing.utils import drop_columns_empty_or_constant


class Scaler:
    def __init__(
        self,
        conf: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        self.conf = conf

    @property
    def columns(self) -> list:
        columns = []
        if self.conf is not None:
            columns.extend(self.conf.keys())
        return columns

    def fit(
        self,
        df: pl.DataFrame,
        metadata: Metadata,
    ) -> None:
        if metadata.targets_regression is None:
            columns_targets = []
        else:
            columns_targets = list(metadata.targets_regression)

        if metadata.features_numeric is None:
            columns_features = []
        else:
            columns_features = list(metadata.features_numeric)

        df = df[columns_targets + columns_features]
        df = drop_columns_empty_or_constant(df=df)
        columns_targets = [column for column in columns_targets if column in df.columns]
        columns_features = [
            column for column in columns_features if column in df.columns
        ]

        conf = {}
        for column in columns_targets + columns_features:
            series = df[column]
            mean = series.mean()
            series = series - mean
            if column in columns_features:
                series = series.fill_nan(0).fill_null(0)
            std = series.std()
            conf[column] = {"mean": mean, "std": std}
        self.conf = conf

    def transform(
        self,
        df: pl.DataFrame,
    ) -> pl.DataFrame:
        exprs = []
        for column in self.conf:
            item = self.conf[column]
            exprs.append((pl.col(column) - item["mean"]) / item["std"])
        df = df.with_columns(*exprs)
        return df

    def fit_transform(
        self,
        df: pl.DataFrame,
        metadata: Metadata,
    ) -> pl.DataFrame:
        self.fit(df=df, metadata=metadata)
        return self.transform(df=df)
