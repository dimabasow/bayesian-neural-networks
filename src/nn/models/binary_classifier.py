from typing import Any, Dict, Literal, Optional, Sequence, Union

import torch

from src.nn.base.bayesian_neural_network import BayesianNeuralNetwork
from src.nn.layers.perceptrone import Perceptrone
from src.nn.layers.res_net import ResNet


class BinaryClassifier(BayesianNeuralNetwork):
    def __init__(
        self,
        dim_in: int,
        dims_hidden: Sequence[int],
        f_act: Union[
            Literal["ELU"],
            Literal["ReLU"],
            Literal["LeakyReLU"],
        ] = "LeakyReLU",
        f_act_kwargs: Optional[Dict[str, Any]] = None,
        backbone: Union[
            Literal["Perceptrone"],
            Literal["ResNet"],
        ] = "Perceptrone",
        lr: float = 0.001,
    ):
        super().__init__()
        self.lr = lr
        if backbone == "Perceptrone":
            self.backbone = Perceptrone(
                dim_in=dim_in,
                dim_out=1,
                dims_hidden=dims_hidden,
                f_act=f_act,
                f_act_kwargs=f_act_kwargs,
            )
        elif backbone == "ResNet":
            self.backbone = ResNet(
                dim_in=dim_in,
                dim_out=1,
                dims_hidden=dims_hidden,
                f_act=f_act,
                f_act_kwargs=f_act_kwargs,
            )
        self.loss_fn = torch.nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def configure_optimizer(
        self, optimizer: str = "Adam", kwargs: Optional[Dict[str, Any]] = None
    ):
        if kwargs is None:
            kwargs = {}
        if "lr" not in kwargs:
            kwargs["lr"] = self.lr
        if "weight_decay" not in kwargs:
            kwargs["weight_decay"] = 0
        return getattr(torch.optim, optimizer)(self.parameters(), **kwargs)

    def negative_likelihood(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        y_pred = self(x)
        return self.loss_fn(y_pred, y)
