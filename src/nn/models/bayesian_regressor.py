from typing import Literal, Dict, Optional, Any, Union, Sequence
import torch
from src.nn import (
    BayesianNeuralNetwork,
    BayesianPerceptrone,
    BayesianResNet,
    BayesianResNetLast,
)


class BayesianRegressor(BayesianNeuralNetwork):
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
        batch_penalty: bool = False,
        batch_affine: bool = True,
        batch_momentum: Optional[float] = 0.1,
        eps: float = 1e-5,
        backbone: Union[
            Literal["Perceptrone"],
            Literal["ResNet"],
            Literal["ResNetLast"],
        ] = "Perceptrone",
        lr: float = 0.001
    ):
        super().__init__()
        self.lr = lr
        if backbone == "Perceptrone":
            self.backbone = BayesianPerceptrone(
                dim_in=dim_in,
                dim_out=2,
                dims_hidden=dims_hidden,
                f_act=f_act,
                f_act_kwargs=f_act_kwargs,
                batch_norm=batch_norm,
                batch_penalty=batch_penalty,
                batch_affine=batch_affine,
                batch_momentum=batch_momentum,
                eps=eps,
            )
        elif backbone == "ResNet":
            self.backbone = BayesianResNet(
                dim_in=dim_in,
                dim_out=2,
                dims_hidden=dims_hidden,
                f_act=f_act,
                f_act_kwargs=f_act_kwargs,
                batch_norm=batch_norm,
                batch_penalty=batch_penalty,
                batch_affine=batch_affine,
                batch_momentum=batch_momentum,
                eps=eps,
            )
        elif backbone == "ResNetLast":
            self.backbone = BayesianResNetLast(
                dim_in=dim_in,
                dim_out=2,
                dims_hidden=dims_hidden,
                f_act=f_act,
                f_act_kwargs=f_act_kwargs,
                batch_norm=batch_norm,
                batch_penalty=batch_penalty,
                batch_affine=batch_affine,
                batch_momentum=batch_momentum,
                eps=eps,
            )
        self.loss_fn = torch.nn.GaussianNLLLoss(full=True, reduction="mean")

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
        y_true = y.view(-1)
        y_pred = self(x).view(-1, 2)
        y_pred_mean = y_pred[:, 0]
        y_pred_var = self.softplus(y_pred[:, 1]) ** 2
        return self.loss_fn(
            y_pred_mean,
            y_true,
            y_pred_var,
        )
