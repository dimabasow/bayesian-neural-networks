from abc import ABC, abstractmethod
import torch


class BayesianModule(torch.nn.Module, ABC):
    @property
    @abstractmethod
    def kl(self) -> torch.Tensor:
        ...

    @property
    @abstractmethod
    def device(self) -> torch.types.Device:
        ...
