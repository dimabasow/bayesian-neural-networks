import copy

import torch

from src.data.abstract_tabular_dataset import AbstractTabularDataset
from src.data.types import PolarsTableItem, TorchTableItem, TorchTargetItem


class TorchTabularDataset(AbstractTabularDataset):
    def make_data(self, data: PolarsTableItem) -> TorchTableItem:
        index = data.index

        if data.features_numeric is None:
            features_numeric = None
        else:
            features_numeric = data.features_numeric.to_torch()

        if data.features_category is None:
            features_category = None
        else:
            features_category = {}
            for name in data.features_category:
                item = data.features_category[name].to_torch().long()
                features_category[name] = item

        if data.target is None:
            target = None
        else:
            target = {}
            for name in data.target:
                target_metadata = self.metadata[name]
                value = data.target[name].value.to_torch()
                if target_metadata.type in {"numeric", "binary"}:
                    value = value.float()
                elif target_metadata.type == "category":
                    value = value.long()

                if data.target[name].mask is None:
                    mask = None
                else:
                    mask = data.target[name].mask.to_torch().bool()
                target[name] = TorchTargetItem(value=value, mask=mask)

        return TorchTableItem(
            index=index,
            features_numeric=features_numeric,
            features_category=features_category,
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

        if data.features_category is None:
            features_category = None
        else:
            features_category = {
                name: data.features_category[name].to(device=device)
                for name in data.features_category
            }

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
            features_category=features_category,
            target=target,
        )
        return dataset

    def cpu(self) -> "TorchTabularDataset":
        return self.to(device="cpu")

    def cuda(self) -> "TorchTabularDataset":
        return self.to(device="cuda")
