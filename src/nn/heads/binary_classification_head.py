from typing import Optional

import torch
from torch.nn.functional import binary_cross_entropy_with_logits, sigmoid

from src.nn.base.bayesian_neural_network_head import BayesianNeuralNetworkHead


class BinaryClassificationHead(BayesianNeuralNetworkHead):
    def negative_likelihood(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return binary_cross_entropy_with_logits(
            input=x,
            target=y.view(-1, 1),
            weight=mask,
            reduction="mean",
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return sigmoid(input=x)
