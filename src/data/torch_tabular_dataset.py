import copy
from typing import Dict, Optional

import polars as pl
import torch

from src.data.abstract_tabular_dataset import (
    AbstractTabularDataset,
    PolarsTableItem,
    TableItem,
    TargetItem,
)


class TorchTargetItem(TargetItem):
    value: torch.Tensor
    mask: Optional[torch.BoolTensor]


class TorchTableItem(TableItem):
    index: Optional[pl.DataFrame]
    features_numeric: Optional[torch.Tensor]
    target: Optional[Dict[str, TorchTargetItem]]


class TorchTabularDataset(AbstractTabularDataset):
    def prepare_data(self, data: PolarsTableItem) -> TorchTableItem:
        index = data.index

        if data.features_numeric is None:
            features_numeric = None
        else:
            features_numeric = data.features_numeric.to_torch().float()

        if data.target is None:
            target = None
        else:
            target = {}

            for name in data.target:
                value = data.target[name].value.to_torch()
                if (
                    self.metadata.targets_multiclass is not None
                    and name in self.metadata.targets_multiclass
                ):
                    value = value.long()
                else:
                    value = value.float()
                if data.target[name].mask is None:
                    mask = None
                else:
                    mask = data.target[name].mask.to_torch().bool()
                target[name] = TorchTargetItem(value=value, mask=mask)

        return TorchTableItem(
            index=index,
            features_numeric=features_numeric,
            target=target,
        )

    def __get_tensor(self) -> torch.Tensor:
        data = self.data
        if data.features_numeric is not None:
            return data.features_numeric
        elif data.features_category is not None:
            return next(iter(data.features_category.values()))
        elif data.target is not None:
            target_item = next(iter(data.target.values()))
            return target_item.value

    @property
    def device(self) -> torch.device:
        return self.__get_tensor().device

    @property
    def is_cpu(self) -> bool:
        return self.__get_tensor().is_cpu

    @property
    def is_cuda(self) -> bool:
        return self.__get_tensor().is_cuda

    def to(self, device: torch.device) -> "TorchTabularDataset":
        dataset = copy.deepcopy(self)
        data: TorchTableItem = dataset.data

        index = data.index

        if data.features_numeric is None:
            features_numeric = None
        else:
            features_numeric = data.features_numeric.to(device=device)

        if data.target is None:
            target = None
        else:
            target = {}
            for name in data.target:
                value = data.target[name].value.to(device=device)
                if data.target[name].mask is None:
                    mask = None
                else:
                    mask = data.target[name].mask.to(device=device)
                target[name] = TorchTargetItem(value=value, mask=mask)

        dataset.data = TorchTableItem(
            index=index,
            features_numeric=features_numeric,
            target=target,
        )
        return dataset

    def cpu(self) -> "TorchTabularDataset":
        return self.to(device="cpu")

    def cuda(self) -> "TorchTabularDataset":
        return self.to(device="cuda")
