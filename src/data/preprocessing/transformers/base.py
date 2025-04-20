import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import polars as pl

from src.data.preprocessing.metadata import TransformType


class BaseTransformer(ABC):
    columns_in: Optional[List[str]] = None
    columns_out: Optional[List[str]]
    transform_type: TransformType

    @abstractmethod
    def __init__(self):
        pass

    @property
    @abstractmethod
    def columns_out(self) -> Optional[List[str]]:
        pass

    @classmethod
    @abstractmethod
    def from_config(
        cls,
        columns: Optional[List[str]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> "BaseTransformer":
        pass

    @property
    @abstractmethod
    def state(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def fit(self, data: pl.DataFrame):
        pass

    @abstractmethod
    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        pass

    @staticmethod
    @abstractmethod
    def filter_raw_data(data: pl.DataFrame) -> pl.DataFrame:
        pass

    @classmethod
    def from_file(cls, path: str) -> "BaseTransformer":
        with open(path, "r") as f:
            state = json.load(f)
        return cls(**state)

    @classmethod
    def from_json(cls, json_str: str) -> "BaseTransformer":
        state = json.loads(json_str)
        return cls(**state)

    def to_file(self, path: str):
        with open(path, "w") as f:
            json.dump(self.state, f, indent=4)

    def to_json(self) -> str:
        return json.dumps(self.state)

    def fit_transform(self, data: pl.DataFrame) -> pl.DataFrame:
        self.fit(data=data)
        return self.transform(data=data)

    def update_columns_in(self, data: pl.DataFrame):
        data = self.filter_raw_data(data=data)
        if self.columns_in is None:
            self.columns_in = list(data.columns)
        else:
            columns_in_set = set(self.columns_in)
            self.columns_in = [
                column for column in data.columns if column in columns_in_set
            ]
