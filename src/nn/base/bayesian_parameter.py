from typing import Iterator
import torch
import torch.types
from src.nn.base import BayesianModule


class BayesianParameter(BayesianModule):
    def __init__(
        self,
        size: tuple[int, ...],
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.rho = torch.nn.Parameter(
            torch.empty(
                size=size,
                **factory_kwargs
            )
        )
        self.gamma = torch.nn.Parameter(
            torch.empty(
                size=size,
                **factory_kwargs
            )
        )

        self.init_parameters = torch.nn.Parameter(
            data=torch.tensor(
                data=[0.0, 1.0, 0.0, 1.0],
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

    def reset_parameters(self) -> None:
        torch.nn.init.normal_(
            tensor=self.rho,
            mean=self.init_parameters[0].item(),
            std=self.softplus(self.init_parameters[1]).item(),
        )
        torch.nn.init.normal_(
            tensor=self.gamma,
            mean=self.init_parameters[2].item(),
            std=self.softplus(self.init_parameters[3]).item(),
        )
        self.init_mode_off()

    def init_mode_on(self):
        self.get_rho = self.get_rho_init_on
        self.get_gamma = self.get_gamma_init_on

    def init_mode_off(self):
        self.get_rho = self.get_rho_init_off
        self.get_gamma = self.get_gamma_init_off

    def get_rho_init_off(self) -> Iterator[torch.nn.Parameter]:
        yield self.rho

    def get_rho_init_on(self) -> Iterator[torch.Tensor]:
        yield torch.normal(
            mean=0,
            std=1,
            size=self.size,
            dtype=self.dtype,
            device=self.device,
            requires_grad=True,
        ) * self.softplus(self.init_parameters[1]) + self.init_parameters[0]

    def get_gamma_init_off(self) -> Iterator[torch.nn.Parameter]:
        yield self.gamma

    def get_gamma_init_on(self) -> Iterator[torch.Tensor]:
        yield torch.normal(
            mean=0,
            std=1,
            size=self.size,
            dtype=self.dtype,
            device=self.device,
            requires_grad=True,
        ) * self.softplus(self.init_parameters[3]) + self.init_parameters[2]

    def get_kl(self) -> torch.Tensor:
        return self.softplus(2 * next(self.get_rho())).sum() / 2

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
