from abc import ABC
from typing import Dict, List, Sequence

import torch

from src.data.torch_tabular_dataset import TorchTargetItem
from src.nn.base.bayesian_module import BayesianModule
from src.nn.base.bayesian_neural_network_backbone import BayesianNeuralNetworkBackbone
from src.nn.base.bayesian_neural_network_head import BayesianNeuralNetworkHead


class BayesianNeuralNetwork(BayesianModule, ABC):
    backbones: Sequence[BayesianNeuralNetworkBackbone]
    heads: Sequence[BayesianNeuralNetworkHead]

    def forward_backbone(
        self,
        x: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        x = x.copy()
        for backbone in self.backbones:
            x[backbone.name_out] = backbone(x=x)
        return x

    def forward(
        self,
        features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        x = {None: features}
        x = self.forward_backbone(x=x)
        y = {}
        for head in self.heads:
            y[head.name_out] = head(x=x[head.name_in])
        return y

    def loss(
        self,
        features: torch.Tensor,
        target: Dict[str, TorchTargetItem],
        train_size: int,
    ) -> torch.Tensor:
        x = {None: features}
        x = self.forward_backbone(x=x)
        loss = torch.zeros(size=[], dtype=self.dtype, device=self.device)
        for head in self.heads:
            loss += head.negative_likelihood(
                x=x[head.name_in],
                y=target[head.name_target].value,
                mask=target[head.name_target].mask,
            )
        loss += self.get_kl() / train_size
        return loss

    def init(
        self,
        features: torch.Tensor,
        optimizer: str = "Adam",
        lr: float = 0.1,
        num_epoch: int = 10,
    ) -> List[float]:
        self.init_mode_on()
        self.train()
        optimizer: torch.optim.Optimizer = getattr(torch.optim, optimizer)(
            self.parameters(),
            lr=lr,
            weight_decay=0,
        )
        metrics = []
        x = {None: features}
        for _ in range(num_epoch):
            optimizer.zero_grad()
            self.forward_backbone(x)
            kl = self.get_kl()
            kl.backward()
            optimizer.step()
            metrics.append(kl.detach().item())
        self.init_mode_off()
        return metrics
