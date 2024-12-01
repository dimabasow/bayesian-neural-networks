from typing import Literal, Dict, Optional, Any, Union
from collections.abc import Sequence
import torch
from src.nn import BayesianNeuralNetwork, BayesianPerceptrone, BayesianResNet


class BayesianBinaryClassifier(BayesianNeuralNetwork):
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
        batch_norm: bool = True,
        batch_penalty: bool = True,
        backbone: Union[
            Literal["Perceptrone"],
            Literal["ResNet"],
        ] = "Perceptrone",
        lr: float = 0.001
    ):
        super().__init__()
        self.lr = lr
        if backbone == "Perceptrone":
            self.backbone = BayesianPerceptrone(
                dim_in=dim_in,
                dim_out=1,
                dims_hidden=dims_hidden,
                f_act=f_act,
                f_act_kwargs=f_act_kwargs,
                batch_norm=batch_norm,
                batch_penalty=batch_penalty,
            )
        elif backbone == "ResNet":
            self.backbone = BayesianResNet(
                dim_in=dim_in,
                dim_out=1,
                dims_hidden=dims_hidden,
                f_act=f_act,
                f_act_kwargs=f_act_kwargs,
                batch_norm=batch_norm,
                batch_penalty=batch_penalty,
            )
        self.loss_fn = torch.nn.BCEWithLogitsLoss(reduction="mean")
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def configure_optimizer(
        self,
        optimizer: str = "Adam",
        kwargs: Optional[Dict[str, Any]] = None
    ):
        if kwargs is None:
            kwargs = {}
        if "lr" not in kwargs:
            kwargs["lr"] = self.lr
        if "weight_decay" not in kwargs:
            kwargs["weight_decay"] = 0
        return getattr(torch.optim, optimizer)(
            self.parameters(),
            **kwargs
        )

    def negative_likelihood(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        y_pred = self(x)
        return self.loss_fn(y_pred, y)
