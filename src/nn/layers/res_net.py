from typing import Literal
import torch
from src.nn.base import BayesianModule


class ResNet(BayesianModule):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        dim_hidden: int,
        n_layers: int,
        f_act: (
            Literal["ELU"]
            | Literal["ReLU"]
            | Literal["LeakyReLU"]
        ) = "LeakyReLU",
    ) -> None:
        super().__init__()

        self.n_layers = n_layers
        self.weights = torch.nn.ModuleDict()
        if f_act == "ELU":
            self.f_act = torch.nn.ELU()
        elif f_act == "ReLU":
            self.f_act = torch.nn.ReLU()
        elif f_act == "LeakyReLU":
            self.f_act = torch.nn.LeakyReLU(negative_slope=3)
        else:
            raise NotImplementedError(
                f"Функция активации {f_act} не импелементирована"
            )
        for i in range(n_layers + 1):
            i += 1
            for k in range(i):
                if k == 0:
                    in_features = dim_in
                else:
                    in_features = dim_hidden
                if i == n_layers + 1:
                    out_features = dim_out
                else:
                    out_features = dim_hidden
                self.weights[f"w_{k}_{i}"] = torch.nn.Linear(
                    in_features=in_features,
                    out_features=out_features,
                    bias=True,
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = [x]
        for i in range(self.n_layers + 1):
            i += 1
            for k in range(i):
                if k == 0:
                    value = self.weights[f"w_{k}_{i}"](z[k])
                else:
                    value = value + self.weights[f"w_{k}_{i}"](z[k])
                if k == i-1:
                    if i != self.n_layers + 1:
                        value = self.f_act(value)
                    z.append(value)
        return z[-1]
