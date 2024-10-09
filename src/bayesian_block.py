import torch
import torch.types
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
        self.softplus = torch.nn.Softplus()

    def forward(self, n: int) -> torch.Tensor:
        noise = torch.normal(
            mean=0,
            std=1,
            size=[n] + list(self.size),
            dtype=self.dtype,
            device=self.device,
        )
        w = self.sigma * (self.gamma + noise)
        return w

    @property
    def size(self) -> torch.Size:
        return self.rho.shape

    def get_sigma(self) -> torch.Tensor:
        return self.softplus(self.rho)

    def get_kl(self) -> torch.Tensor:
        gamma = self.gamma
        gama_pow_2 = gamma**2
        nu = torch.log(1 + gama_pow_2)
        nu_pow_2 = nu**2
        return nu_pow_2.sum() / 2
