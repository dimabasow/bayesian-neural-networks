from abc import ABC, abstractmethod
import torch


class BayesianModule(torch.nn.Module, ABC):
    @property
    @abstractmethod
    def kl(self) -> torch.Tensor:
        ...

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype
