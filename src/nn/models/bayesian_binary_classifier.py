from typing import Literal
import torch
from src.nn import BayesianNeuralNetwork, BayesianResNet


class BayesianBincaryClassifier(BayesianNeuralNetwork):
    def __init__(
        self,
        dim_in: int,
        dim_hidden: int,
        n_layers: int,
        f_act: Literal["elu"] | Literal["relu"] = "elu",
        lr: float = 0.001
    ):
        super().__init__()
        self.lr = lr
        self.resnet = BayesianResNet(
            dim_in=dim_in,
            dim_out=1,
            dim_hidden=dim_hidden,
            n_layers=n_layers,
            f_act=f_act,
        )
        self.scale = torch.nn.Parameter(
            torch.ones(size=(dim_in,))
        )
        self.shift = torch.nn.Parameter(
            torch.zeros(size=(dim_in,))
        )
        self.loss_fn = torch.nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet((x + self.shift) * self.scale)

    def configure_optimizer(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=0,
        )
        return optimizer

    def negative_likelihood(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        y_pred = self(x)
        return self.loss_fn(y_pred, y)
