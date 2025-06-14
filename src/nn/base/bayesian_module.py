from abc import ABC
from typing import Iterator, Optional, Sequence

import torch


class BayesianModule(torch.nn.Module, ABC):
    softplus = torch.nn.Softplus()
    is_init_mode_on: bool = True

    def bayesian_modules(self) -> Iterator["BayesianModule"]:
        for name in self._modules:
            module = self._modules[name]
            if isinstance(module, BayesianModule):
                yield module

    def get_init_parameters(self) -> Iterator[torch.Tensor]:
        for module in self.bayesian_modules():
            for parameter in module.get_init_parameters():
                yield parameter

    def init_mode_on(self):
        self.is_init_mode_on = True
        for module in self.bayesian_modules():
            module.init_mode_on()

    def init_mode_off(self):
        self.is_init_mode_on = False
        for module in self.bayesian_modules():
            module.init_mode_off()

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
            for gamma, rho in zip(module.get_gamma(), module.get_rho()):
                yield gamma * self.softplus(rho)

    def get_kl(self) -> torch.Tensor:
        kl = torch.zeros(size=[], dtype=self.dtype, device=self.device)
        for module in self.bayesian_modules():
            kl += module.get_kl()
        return kl

    @property
    def device(self) -> torch.device:
        for parameter in self.parameters():
            return parameter.device
        for module in self.bayesian_modules():
            return module.device

    @property
    def dtype(self) -> torch.dtype:
        for parameter in self.parameters():
            return parameter.dtype
        for module in self.bayesian_modules():
            return module.dtype

    def sample_noise(
        self,
        dim_batch: Optional[Sequence[int]] = None,
        dim_expand: Optional[Sequence[int]] = None,
        dim_sample: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        if dim_batch is None:
            dim_batch = tuple()
        if dim_sample is None:
            dim_sample = tuple()
        if dim_expand is None:
            dim_expand = tuple()

        noise = torch.normal(
            mean=0,
            std=1,
            size=list(dim_batch) + list(dim_sample),
            dtype=self.dtype,
            device=self.device,
        )

        noise = noise.reshape(
            *dim_batch,
            *([1] * len(dim_expand)),
            *dim_sample,
        )
        noise = noise.expand(
            *dim_batch,
            *dim_expand,
            *dim_sample,
        )
        return noise
