from typing import Any, Dict, Literal, Optional, Sequence, Union

import torch

from src.nn.affine import BayesianAffine
from src.nn.base import BayesianModule
from src.nn.batchnorm import BayesianBatchNorm
from src.nn.container import BayesianSequential
from src.nn.linear import BayesianLinear


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
        batch_penalty: bool = False,
        batch_affine: bool = True,
        batch_momentum: Optional[float] = 0.1,
        eps: float = 1e-5,
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
                    momentum=batch_momentum,
                    eps=eps,
                )
            )
            if batch_affine:
                self.fcc.append(BayesianAffine(size=[dim_hidden]))
            self.fcc.append(getattr(torch.nn, f_act)(**f_act_kwargs))
            in_features = dim_hidden
        self.fcc.append(
            BayesianLinear(
                in_features=in_features,
                out_features=dim_out,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fcc(x)
