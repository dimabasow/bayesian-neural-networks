from typing import Literal
import torch
from src.nn.base import BayesianModule
from src.nn.layers import BayesianLinear
from src.nn.container import BayesianSequential


class BayesianPerceptrone(BayesianModule):
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
        self.fcc = BayesianSequential()
        in_features = dim_in
        for _ in range(n_layers):
            self.fcc.append(
                BayesianLinear(
                    in_features=in_features,
                    out_features=dim_hidden,
                )
            )
            in_features = dim_hidden
            if f_act == "elu":
                self.fcc.append(torch.nn.ELU())
            elif f_act == "relu":
                self.fcc.append(torch.nn.ReLU())
            else:
                raise NotImplementedError(
                    f"Функция активации {f_act} не импелементирована"
                )
        self.fcc.append(
            BayesianLinear(
                in_features=in_features,
                out_features=dim_out,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fcc(x)
