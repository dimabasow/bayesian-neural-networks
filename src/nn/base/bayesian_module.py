from abc import ABC
from typing import Iterator
import torch
import torch.types


class BayesianModule(torch.nn.Module, ABC):

    softplus = torch.nn.Softplus()

    def bayesian_modules(self) -> Iterator["BayesianModule"]:
        for name in self._modules:
            module = self._modules[name]
            if isinstance(module, BayesianModule):
                yield module

    def reset_parameters(self) -> None:
        for module in self.bayesian_modules():
            module.reset_parameters()

    def get_rho(self) -> Iterator[torch.nn.Parameter]:
        for module in self.bayesian_modules():
            for rho in module.get_rho():
                yield rho

    def get_gamma(self) -> Iterator[torch.nn.Parameter]:
        for module in self.bayesian_modules():
            for gamma in module.get_gamma():
                yield gamma

    def get_sigma(self) -> Iterator[torch.nn.Parameter]:
        for module in self.bayesian_modules():
            for rho in module.get_rho():
                yield self.softplus(rho)

    def get_mu(self) -> Iterator[torch.nn.Parameter]:
        for module in self.bayesian_modules():
            for gamma, sigma in zip(module.get_gamma(), module.get_sigma()):
                yield gamma * sigma

    def get_kl(self) -> torch.Tensor:
        kl = torch.zeros(size=[], dtype=self.dtype, device=self.device)
        for module in self.bayesian_modules():
            for gamma in module.get_gamma():
                gamma_pow_2 = gamma**2
                nu = torch.log(1 + gamma_pow_2).view(-1)
                kl += torch.dot(nu, nu) / 2
        return kl

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype
