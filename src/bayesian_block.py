import torch
from .bayesian_module import BayesianModule


class BayesianBlock(BayesianModule):
    def __init__(
        self, size: tuple[int, ...],
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.rho = torch.nn.Parameter(
            torch.ones(size=size, **factory_kwargs)
        )
        self.gamma = torch.nn.Parameter(
            torch.zeros(size=size, **factory_kwargs)
        )
        self.softplus = torch.nn.Softplus()

    def reset_parameters(self) -> None:
        self.rho = self.rho.new_ones(size=self.rho.shape)
        self.gamma = self.gamma.new_zeros(size=self.gamma.shape)

    def forward(self, n: int) -> torch.Tensor:
        w = torch.normal(
            mean=0,
            std=1,
            size=[n] + list(self.size)
        )
        w = w*self.sigma + self.mu
        return w

    @property
    def size(self) -> torch.Size:
        return self.rho.shape

    @property
    def sigma(self) -> torch.Tensor:
        return self.softplus(self.rho)

    @property
    def mu(self) -> torch.Tensor:
        return self.gamma * self.rho

    @property
    def nu(self) -> torch.Tensor:
        return torch.log(1 + self.gamma*self.gamma)

    @property
    def kl(self) -> torch.Tensor:
        nu = self.nu
        return (nu*nu).sum() / 2

    @property
    def w(self) -> torch.Tensor:
        w = torch.normal(
            mean=0,
            std=1,
            size=self.size
        )
        w = w*self.sigma + self.mu
        return w
