from typing import Optional

import torch
from torch.nn.functional import gaussian_nll_loss

from src.nn.base.bayesian_neural_network_head import BayesianNeuralNetworkHead


class GaussianRegressionHead(BayesianNeuralNetworkHead):

    def __init__(self, name_in, name_out, name_target):
        super().__init__(name_in, name_out, name_target)
        self.sigma = torch.nn.Parameter(
            data=torch.tensor(
                data=1,
                dtype=torch.float32,
            )
        )

    def negative_likelihood(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        dim_batch_expand_shape = x.shape[:-1]
        mu = x.view(*dim_batch_expand_shape)
        nll = gaussian_nll_loss(
            input=mu,
            target=y,
            var=self.sigma ** 2,
            reduction="none",
            full=True,
        )
        if mask is None:
            return nll.mean()
        else:
            return (nll * mask.view(-1)).mean()

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        dim_batch_expand = x.shape[:-1]
        if self.training:
            dim_batch = dim_batch_expand
            dim_expand = None
        else:
            dim_batch = dim_batch_expand[:1]
            dim_expand = dim_batch_expand[1:]
        noise = self.sample_noise(
            dim_batch=dim_batch,
            dim_expand=dim_expand,
        )
        mu = x.view(*dim_batch_expand)

        return (noise * self.sigma) + mu
