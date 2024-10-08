from typing import Literal
import torch
from src import BayesianLinear, BayesianModule


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
        self.weights = torch.nn.ModuleList()
        if f_act == "elu":
            self.f_act = torch.nn.ELU()
        elif f_act == "relu":
            self.f_act = torch.nn.ReLU()
        else:
            raise NotImplementedError(
                f"Функция активации {f_act} не импелементирована"
            )
        for i in range(n_layers):
            if i == 0:
                in_features = dim_in
            else:
                in_features = dim_hidden
            if i == n_layers - 1:
                out_features = dim_out
            else:
                out_features = dim_hidden
            self.weights.append(
                BayesianLinear(
                    in_features=in_features,
                    out_features=out_features,
                )
            )

        self.fcc = torch.nn.Sequential()
        for item in self.weights:
            self.fcc.append(item)
            if f_act == "elu":
                self.fcc.append(torch.nn.ELU())
            elif f_act == "relu":
                self.fcc.append(torch.nn.ReLU())
            else:
                raise NotImplementedError(
                    f"Функция активации {f_act} не импелементирована"
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fcc(x)

    @property
    def kl(self) -> torch.Tensor:
        for count, item in enumerate(self.weights):
            if count == 0:
                kl = item.kl
            else:
                kl = kl + item.kl
        return kl
