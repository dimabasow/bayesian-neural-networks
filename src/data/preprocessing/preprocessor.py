from typing import (
    NamedTuple,
    Optional,
    Tuple,
    Dict,
    Any,
    Sequence,
    Union,
    List,
)
import json
import polars as pl
import src.data.preprocessing.transformers
from src.data.preprocessing.transformers import BaseTransformer
from src.data.preprocessing.metadata import Metadata


class RuleTransform(NamedTuple):
    transformer: str
    columns: Optional[Tuple[str, ...]] = None
    kwargs: Optional[Dict[str, Any]] = None


class Preprocessor:
    def __init__(
        self,
        transformers: Sequence[BaseTransformer],
    ):
        self.transformers = transformers

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
        return cls(transformers=tuple(transformers))

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
        ids = []
        features_numeric = []
        features_category = []
        targets_regression = []
        targets_binary = []
        targets_multiclass = []
        for item in self.transformers:
            if item.metadata.ids is not None:
                ids += list(item.metadata.ids)
            if item.metadata.features_numeric is not None:
                features_numeric += list(item.metadata.features_numeric)
            if item.metadata.features_category is not None:
                features_category += list(item.metadata.features_category)
            if item.metadata.targets_regression is not None:
                targets_regression += list(item.metadata.targets_regression)
            if item.metadata.targets_binary is not None:
                targets_binary += list(item.metadata.targets_binary)
            if item.metadata.targets_multiclass is not None:
                targets_multiclass += list(item.metadata.targets_multiclass)
        ids = sorted(set(ids))
        features_numeric = sorted(set(features_numeric))
        features_category = sorted(set(features_category))
        targets_regression = sorted(set(targets_regression))
        targets_binary = sorted(set(targets_binary))
        targets_multiclass = sorted(set(targets_multiclass))
        if not ids:
            ids = None
        if not features_numeric:
            features_numeric = None
        if not features_category:
            features_category = None
        if not targets_regression:
            targets_regression = None
        if not targets_binary:
            targets_binary = None
        if not targets_multiclass:
            targets_multiclass = None
        return Metadata(
            ids=tuple(ids),
            features_numeric=tuple(features_numeric),
            features_category=tuple(features_category),
            targets_regression=tuple(targets_regression),
            targets_binary=tuple(targets_binary),
            targets_multiclass=tuple(targets_multiclass),
        )

    @property
    def columns_in(self) -> List[str]:
        columns = []
        for item in self.transformers:
            if item.columns_in is not None:
                columns += list(item.columns_in)
        return columns

    @property
    def columns_out(self) -> List[str]:
        metadata = self.metadata
        return sorted(
            set(metadata.ids)
            | set(metadata.features_numeric)
            | set(metadata.features_category)
            | set(metadata.targets_regression)
            | set(metadata.targets_binary)
            | set(metadata.targets_multiclass)
        )

    @property
    def state(self) -> List[Dict[str, Union[str, Dict]]]:
        return [
            {
                "name": item.__class__.__name__,
                "state": item.state
            }
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
        columns = [
            column
            for column in self.columns_out
            if column in df.columns
        ]
        return df[columns]

    def fit_transform(self, data: pl.DataFrame) -> pl.DataFrame:
        dfs = (item.fit_transform(data=data) for item in self.transformers)
        df = pl.concat(dfs, how="horizontal")
        columns = [
            column
            for column in self.columns_out
            if column in df.columns
        ]
        return df[columns]
