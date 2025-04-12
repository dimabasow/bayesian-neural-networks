from typing import Iterator, Sequence

import torch

from src.nn.base.bayesian_module import BayesianModule


class BayesianParameter(BayesianModule):
    def __init__(
        self,
        size: Sequence[int],
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.gamma = torch.nn.Parameter(
            torch.empty(size=list(size), **factory_kwargs),
        )
        self.rho = torch.nn.Parameter(
            torch.empty(size=list(size), **factory_kwargs),
        )
        self.init_parameters = torch.nn.Parameter(
            data=torch.zeros(size=[4], **factory_kwargs),
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

    def get_init_gamma(self) -> torch.Tensor:
        return (
            torch.randn_like(self.gamma) * self.softplus(self.init_parameters[0])
            + self.init_parameters[1]
        )

    def get_init_rho(self) -> torch.Tensor:
        return (
            torch.randn_like(self.rho) * self.softplus(self.init_parameters[2])
            + self.init_parameters[3]
        )

    def reset_parameters(self) -> None:
        self.gamma = torch.nn.Parameter(self.get_init_gamma())
        self.rho = torch.nn.Parameter(self.get_init_rho())

    def get_gamma(self) -> Iterator[torch.nn.Parameter]:
        if self.is_init_mode_on:
            yield self.get_init_gamma()
        else:
            yield self.gamma

    def get_rho(self) -> Iterator[torch.nn.Parameter]:
        if self.is_init_mode_on:
            yield self.get_init_rho()
        else:
            yield self.rho

    def get_kl(self) -> torch.Tensor:
        gamma = next(self.get_gamma())
        gamma_pow_2 = gamma**2
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
