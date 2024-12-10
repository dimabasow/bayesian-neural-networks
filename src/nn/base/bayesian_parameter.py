from typing import Iterator, Sequence
import torch
import torch.types
from src.nn.base import BayesianModule


class BayesianParameter(BayesianModule):
    def __init__(
        self,
        size: Sequence[int],
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.rho = torch.nn.Parameter(
            torch.empty(
                size=list(size),
                requires_grad=True,
                **factory_kwargs
            ),
            requires_grad=True,
        )
        self.gamma = torch.nn.Parameter(
            torch.empty(
                size=list(size),
                requires_grad=True,
                **factory_kwargs
            ),
            requires_grad=True,
        )
        self.register_buffer(
            "init_gamma",
            torch.randn(
                size=self.size,
                requires_grad=True
            )
        )
        self.init_gamma: torch.Tensor
        self.register_buffer(
            "init_rho",
            torch.randn(
                size=self.size,
                requires_grad=True
            )
        )
        self.init_rho: torch.Tensor
        self.init_parameters = torch.nn.Parameter(
            data=torch.tensor(
                data=[
                    torch.zeros(size=[]),
                    torch.ones(size=[]),
                    torch.zeros(size=[]),
                    torch.ones(size=[]),
                ],
                requires_grad=True,
                **factory_kwargs
            ),
            requires_grad=True,
        )
        self.reset_parameters()

    @property
    def size(self) -> torch.Size:
        return self.rho.shape

    def bayesian_modules(self) -> Iterator[BayesianModule]:
        yield self

    def get_init_parameters(self) -> Iterator[torch.Tensor]:
        yield self.init_parameters

    def init_mode_on(self):
        self.is_init_mode_on = True

    def init_mode_off(self):
        self.is_init_mode_on = False
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.gamma = torch.nn.Parameter(
            (
                self.init_gamma * self.init_parameters[1]
                + self.init_parameters[0]
            ),
            requires_grad=True,
        )
        self.rho = torch.nn.Parameter(
            (
                self.init_rho * self.init_parameters[3]
                + self.init_parameters[2]
            ),
            requires_grad=True,
        )

    def get_gamma(self) -> Iterator[torch.nn.Parameter]:
        if self.is_init_mode_on:
            yield (
                self.init_gamma * self.init_parameters[1]
                + self.init_parameters[0]
            )
        else:
            yield self.gamma

    def get_rho(self) -> Iterator[torch.nn.Parameter]:
        if self.is_init_mode_on:
            yield (
                self.init_rho * self.init_parameters[3]
                + self.init_parameters[2]
            )
        else:
            yield self.rho

    def get_kl(self) -> torch.Tensor:
        gamma = next(self.get_gamma())
        gamma_pow_2 = gamma ** 2
        kl = (torch.log(1 + gamma_pow_2)).sum() / 2
        return kl

    def forward(self, *dim: int) -> torch.Tensor:
        noise = torch.normal(
            mean=0,
            std=1,
            size=list(dim) + list(self.size),
            dtype=self.dtype,
            device=self.device,
        )
        sigma = next(self.get_sigma())
        gamma = next(self.get_gamma())
        mu = gamma * sigma
        w = (noise * sigma) + mu
        return w
