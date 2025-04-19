from typing import Any, Dict, Literal, Optional, Sequence, Union

import torch
from torch.nn import Linear, Sequential

from src.nn.base.bayesian_neural_network_backbone import BayesianNeuralNetworkBackbone


class Perceptrone(BayesianNeuralNetworkBackbone):
    def __init__(
        self,
        name_in: str,
        name_out: str,
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
        self.name_in = name_in
        self.name_out = name_out

        if f_act_kwargs is None:
            f_act_kwargs = {}
        self.fcc = Sequential()
        in_features = dim_in
        for dim_hidden in dims_hidden:
            self.fcc.append(
                Linear(
                    in_features=in_features,
                    out_features=dim_hidden,
                )
            )
            self.fcc.append(getattr(torch.nn, f_act)(**f_act_kwargs))
            in_features = dim_hidden
        self.fcc.append(
            Linear(
                in_features=in_features,
                out_features=dim_out,
            )
        )

    def forward(
        self,
        x: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return self.fcc(x[self.name_in])
