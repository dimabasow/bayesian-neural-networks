from abc import ABC, abstractmethod
from typing import Dict

import torch

from src.nn.base.bayesian_module import BayesianModule


class BayesianNeuralNetworkBackbone(BayesianModule, ABC):
    name_out: str

    @abstractmethod
    def forward(
        self,
        x: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        pass
