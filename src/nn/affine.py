from typing import Sequence

import torch

from src.nn.base import BayesianModule, BayesianParameter


class BayesianAffine(BayesianModule):
    __constants__ = [
        "size",
    ]
    size: Sequence[int]

    def __init__(
        self,
        size: Sequence[int],
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.size = size
        self.scale = BayesianParameter(size=size, **factory_kwargs)
        self.shift = BayesianParameter(
            size=size,
            **factory_kwargs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * (1 + self.scale) + self.shift
