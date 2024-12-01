from typing import Literal, Union, Optional, Dict, Any
from collections.abc import Sequence
import torch
from src.nn.base import BayesianModule
from src.nn.linear import BayesianLinear
from src.nn.batchnorm import BayesianBatchNorm
from src.nn.container import BayesianSequential


class BayesianPerceptrone(BayesianModule):
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
        batch_norm: bool = True,
        batch_penalty: bool = True,
    ) -> None:
        super().__init__()

        if f_act_kwargs is None:
            f_act_kwargs = {}
        self.fcc = BayesianSequential()
        in_features = dim_in
        for dim_hidden in dims_hidden:
            self.fcc.append(
                BayesianLinear(
                    in_features=in_features,
                    out_features=dim_hidden,
                )
            )
            self.fcc.append(
                BayesianBatchNorm(
                    size=[dim_hidden],
                    transform=batch_norm,
                    penalty=batch_penalty,
                )
            )
            in_features = dim_hidden
            self.fcc.append(
                getattr(torch.nn, f_act)(**f_act_kwargs)
            )
        self.fcc.append(
            BayesianLinear(
                in_features=in_features,
                out_features=dim_out,
            )
        )
        # self.fcc.append(
        #     BayesianBatchNorm(
        #         size=[dim_out],
        #         transform=batch_norm,
        #         penalty=batch_penalty,
        #     )
        # )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fcc(x)
