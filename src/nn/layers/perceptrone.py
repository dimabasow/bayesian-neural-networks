from typing import Literal, Union
import torch
from src.nn.base import BayesianModule


class Perceptrone(BayesianModule):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        dim_hidden: int,
        n_layers: int,
        f_act: Union[
            Literal["ELU"],
            Literal["ReLU"],
            Literal["LeakyReLU"],
        ] = "LeakyReLU",
    ) -> None:
        super().__init__()

        self.n_layers = n_layers
        self.fcc = torch.nn.Sequential()
        in_features = dim_in
        for _ in range(n_layers):
            self.fcc.append(
                torch.nn.Linear(
                    in_features=in_features,
                    out_features=dim_hidden,
                )
            )
            in_features = dim_hidden
            if f_act == "ELU":
                self.fcc.append(torch.nn.ELU())
            elif f_act == "ReLU":
                self.fcc.append(torch.nn.ReLU())
            elif f_act == "LeakyReLU":
                self.fcc.append(torch.nn.LeakyReLU(negative_slope=3))
            else:
                raise NotImplementedError(
                    f"Функция активации {f_act} не импелементирована"
                )
        self.fcc.append(
            torch.nn.Linear(
                in_features=in_features,
                out_features=dim_out,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fcc(x)
