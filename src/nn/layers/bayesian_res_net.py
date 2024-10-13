from typing import Literal
import torch
from src.nn.base import BayesianModule
from src.nn.layers import BayesianLinear
from src.nn.container import BayesianModuleDict


class BayesianResNet(BayesianModule):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        dim_hidden: int,
        n_layers: int,
        f_act: Literal["elu"] | Literal["relu"] = "elu",
    ) -> None:
        super().__init__()

        self.n_layers = n_layers
        self.weights = BayesianModuleDict()
        if f_act == "elu":
            self.f_act = torch.nn.ELU()
        elif f_act == "relu":
            self.f_act = torch.nn.ReLU()
        else:
            raise NotImplementedError(
                f"Функция активации {f_act} не импелементирована"
            )
        for i in range(n_layers):
            for k in range(i):
                if k == 0:
                    in_features = dim_in
                else:
                    in_features = dim_hidden
                if i == n_layers - 1:
                    out_features = dim_out
                else:
                    out_features = dim_hidden
                self.weights[f"w_{k}_{i}"] = BayesianLinear(
                    in_features=in_features,
                    out_features=out_features,
                    bias=True,
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = [x]
        for i in range(self.n_layers):
            for k in range(i):
                if k == 0:
                    value = self.weights[f"w_{k}_{i}"](z[k])
                else:
                    value = value + self.weights[f"w_{k}_{i}"](z[k])
                if k == i-1:
                    if i != self.n_layers - 1:
                        value = self.f_act(value)
                    z.append(value)
        return z[-1]
