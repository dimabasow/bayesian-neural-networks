from typing import Iterator
import torch
from torch.nn.parameter import Parameter as Parameter
import torch.types
from src.nn.base import BayesianModule


class BayesianBlock(BayesianModule):
    def __init__(
        self, size: tuple[int, ...],
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.rho = torch.nn.Parameter(
            torch.ones(
                size=size,
                **factory_kwargs
            )
        )
        self.gamma = torch.nn.Parameter(
            torch.zeros(
                size=size,
                **factory_kwargs
            )
        )

    @property
    def size(self) -> torch.Size:
        return self.rho.shape

    def bayesian_modules(self) -> Iterator[BayesianModule]:
        yield self

    def reset_parameters(self) -> None:
        torch.nn.init.ones_(self.rho)
        torch.nn.init.zeros_(self.gamma)

    def get_rho(self) -> Iterator[Parameter]:
        yield self.rho

    def get_gamma(self) -> Iterator[Parameter]:
        yield self.gamma

    def forward(self, *dim: int) -> torch.Tensor:
        noise = torch.normal(
            mean=0,
            std=1,
            size=list(dim) + list(self.size),
            dtype=self.dtype,
            device=self.device,
        )
        w = self.softplus(self.rho) * (self.gamma + noise)
        return w
