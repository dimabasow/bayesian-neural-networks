from abc import ABC, abstractmethod
from typing import Optional

import torch

from src.nn.base.bayesian_module import BayesianModule


class BayesianNeuralNetworkHead(BayesianModule, ABC):
    name_in: str
    name_out: str
    name_target: str

    def __init__(
        self,
        name_in: str,
        name_out: str,
        name_target: str,
    ) -> None:
        super().__init__()
        self.name_in = name_in
        self.name_out = name_out
        self.name_target = name_target

    @abstractmethod
    def negative_likelihood(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        pass
