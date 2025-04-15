import json
from typing import (
    Any,
    Dict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import polars as pl

import src.data.preprocessing.transformers
from src.data.preprocessing.metadata import Metadata
from src.data.preprocessing.transformers import BaseTransformer


class RuleTransform(NamedTuple):
    transformer: str
    columns: Optional[Tuple[str, ...]] = None
    kwargs: Optional[Dict[str, Any]] = None


class Preprocessor:
    def __init__(
        self,
        transformers: Sequence[BaseTransformer],
    ):
        self.transformers = tuple(transformers)

    @classmethod
    def from_rules(cls, *rules: RuleTransform) -> "Preprocessor":
        transformers = []
        for rule in rules:
            constructor: BaseTransformer = getattr(
                src.data.preprocessing.transformers,
                rule.transformer,
            )
            transformer = constructor.from_config(
                columns=rule.columns,
                kwargs=rule.kwargs,
            )
            transformers.append(transformer)
        return cls(transformers=transformers)

    @classmethod
    def from_state(
        cls,
        state: Sequence[Dict[str, Union[str, Dict]]],
    ) -> "Preprocessor":
        transformers = []
        for item in state:
            cls_name = item["name"]
            cls_state = item["state"]
            constructor: BaseTransformer = getattr(
                src.data.preprocessing.transformers,
                cls_name,
            )
            transformer = constructor(**cls_state)
            transformers.append(transformer)
        return cls(transformers=tuple(transformers))

    @classmethod
    def from_json(cls, json_str: str) -> "Preprocessor":
        state = json.loads(json_str)
        return cls(**state)

    @property
    def metadata(self) -> Metadata:
        metadata_dict: Dict[str, List[str]] = {}
        for item in self.transformers:
            transform_type = item.transform_type.name
            if transform_type not in metadata_dict:
                metadata_dict[transform_type] = []
            metadata_dict[transform_type].extend(item.columns_out)

        for key, value in metadata_dict.items():
            metadata_dict[key] = sorted(set(value))
        return Metadata(**metadata_dict)

    @property
    def columns_in(self) -> List[str]:
        columns = []
        for item in self.transformers:
            if item.columns_in is not None:
                columns += list(item.columns_in)
        return sorted(set(columns))

    @property
    def columns_out(self) -> List[str]:
        dict_metadata = self.metadata._asdict()

        columns = []
        for key in [
            "index",
            "targets_regression",
            "targets_binary",
            "targets_multiclass",
            "features_numeric",
        ]:
            value = dict_metadata[key]
            if value is not None:
                columns.extend(value)
        return columns

    @property
    def state(self) -> List[Dict[str, Union[str, Dict]]]:
        return [
            {"name": item.__class__.__name__, "state": item.state}
            for item in self.transformers
        ]

    def to_json(self) -> str:
        return json.dumps(self.state)

    def fit(self, data: pl.DataFrame):
        for item in self.transformers:
            item.fit(data=data)

    def transform(self, data: pl.DataFrame) -> pl.DataFrame:
        dfs = (item.transform(data=data) for item in self.transformers)
        df = pl.concat(dfs, how="horizontal")
        columns = [column for column in self.columns_out if column in df.columns]
        return df[columns]

    def fit_transform(self, data: pl.DataFrame) -> pl.DataFrame:
        dfs = (item.fit_transform(data=data) for item in self.transformers)
        df = pl.concat(dfs, how="horizontal")
        columns = [column for column in self.columns_out if column in df.columns]
        return df[columns]
