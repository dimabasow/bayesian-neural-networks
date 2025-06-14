from typing import Dict

import torch

from src.nn.base.bayesian_neural_network_backbone import BayesianNeuralNetworkBackbone
from src.nn.linear import BayesianLinear


class BayesianLinearBackbone(BayesianNeuralNetworkBackbone):
    def __init__(
        self,
        name_in: str,
        name_out: str,
        dim_in: int,
        dim_out: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.name_in = name_in
        self.name_out = name_out

        self.linear = BayesianLinear(
            in_features=dim_in,
            out_features=dim_out,
            bias=bias,
        )

    def forward(
        self,
        x: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return self.linear(x[self.name_in])
