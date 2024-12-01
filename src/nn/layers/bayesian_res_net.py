from typing import Literal, Union, Dict, Any, Optional
from collections.abc import Sequence
import torch
from src.nn.base import BayesianModule
from src.nn.linear import BayesianLinear
from src.nn.container import BayesianModuleDict, BayesianModuleList, BayesianSequential
from src.nn.batchnorm import BayesianBatchNorm


class BayesianResNet(BayesianModule):
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
        self.dims_hidden = dims_hidden

        self.weights = BayesianModuleDict()
        dims = [dim_in] + list(dims_hidden) + [dim_out]
        for i, dim_i in enumerate(dims):
            for j, dim_j in enumerate(dims[i+1:]):
                self.weights[f"w_{i}_{i+j+1}"] = BayesianLinear(
                    in_features=dim_i,
                    out_features=dim_j,
                    bias=True
                )

        self.f_act_blocks = BayesianModuleList()
        for dim in dims_hidden:
            block = BayesianSequential(
                BayesianBatchNorm(
                    size=[dim],
                    transform=batch_norm,
                    penalty=batch_penalty,
                ),
                getattr(torch.nn, f_act)(**f_act_kwargs),
            )
            self.f_act_blocks.append(block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = [x]
        for i in range(len(self.dims_hidden) + 1):
            i += 1
            for k in range(i):
                if k == 0:
                    value = self.weights[f"w_{k}_{i}"](z[k])
                else:
                    value = value + self.weights[f"w_{k}_{i}"](z[k])
                if k == i-1:
                    if i != len(self.dims_hidden) + 1:
                        value = self.f_act_blocks[i-1](value)
                    z.append(value)
        return z[-1]
