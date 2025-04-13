from typing import Any, Dict, Literal, Optional, Sequence, Union

import torch

from src.nn.affine import BayesianAffine
from src.nn.base.bayesian_module import BayesianModule
from src.nn.batchnorm import BayesianBatchNorm
from src.nn.container import (
    BayesianModuleDict,
    BayesianModuleList,
    BayesianSequential,
)
from src.nn.linear import BayesianLinear


class BayesianResNetLast(BayesianModule):
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
        self.dims_hidden = dims_hidden

        self.weights = BayesianModuleDict()
        dims = [dim_in] + list(dims_hidden) + [dim_out]
        j_out = len(dims) - 1
        for i, dim_i in enumerate(dims[:-1]):
            j = i + 1
            dim_j = dims[j]
            self.weights[f"w_{i}_{j}"] = BayesianLinear(
                in_features=dim_i,
                out_features=dim_j,
                bias=True,
            )
            if j != j_out:
                self.weights[f"w_{i}_{j_out}"] = BayesianLinear(
                    in_features=dim_i,
                    out_features=dim_out,
                    bias=True,
                )

        self.f_act_blocks = BayesianModuleList()
        for dim in dims_hidden:
            block = BayesianSequential()
            block.append(
                BayesianBatchNorm(
                    size=[dim],
                    transform=batch_norm,
                    penalty=batch_penalty,
                    momentum=batch_momentum,
                    eps=eps,
                )
            )
            if batch_affine:
                block.append(BayesianAffine(size=[dim]))
            block.append(getattr(torch.nn, f_act)(**f_act_kwargs))
            self.f_act_blocks.append(block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = [x]
        for i in range(len(self.dims_hidden)):
            j = i + 1
            value = self.weights[f"w_{i}_{j}"](z[i])
            value = self.f_act_blocks[i](value)
            z.append(value)
        j_out = len(z)
        for k, z_k in enumerate(z):
            if k == 0:
                value = self.weights[f"w_{k}_{j_out}"](z_k)
            else:
                value = value + self.weights[f"w_{k}_{j_out}"](z_k)
        return value
