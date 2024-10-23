from typing import Iterator
import torch
from torch.nn.parameter import Parameter as Parameter
import torch.types
from src.nn.base import BayesianModule


class BayesianParameter(BayesianModule):
    def __init__(
        self, size: tuple[int, ...],
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.rho = torch.nn.Parameter(
            torch.zeros(
                size=size,
                **factory_kwargs
            )
        )
        self.gamma = torch.nn.Parameter(
            torch.normal(
                mean=0,
                std=1,
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
        torch.nn.init.zeros_(self.rho)
        torch.nn.init.normal_(self.gamma, mean=0, std=1)

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
        w = (noise * next(self.get_sigma())) + next(self.get_mu())
        return w
