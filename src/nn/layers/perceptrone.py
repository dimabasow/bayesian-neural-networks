from typing import Literal, Union, Optional, Dict, Any, Sequence
import torch
from src.nn.base import BayesianModule


class Perceptrone(BayesianModule):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        dims_hidden: Sequence[int],
        f_act: Union[
            Literal["ELU"],
            Literal["ReLU"],
            Literal["LeakyReLU"],
        ] = "LeakyReLU",
        f_act_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        if f_act_kwargs is None:
            f_act_kwargs = {}

        self.fcc = torch.nn.Sequential()
        in_features = dim_in
        for dim_hidden in dims_hidden:
            self.fcc.append(
                torch.nn.Linear(
                    in_features=in_features,
                    out_features=dim_hidden,
                )
            )
            self.fcc.append(
                getattr(torch.nn, f_act)(**f_act_kwargs)
            )
            in_features = dim_hidden
        self.fcc.append(
            torch.nn.Linear(
                in_features=in_features,
                out_features=dim_out,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fcc(x)
