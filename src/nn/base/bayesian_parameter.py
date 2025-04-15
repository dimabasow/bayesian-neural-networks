from typing import Iterator, Optional, Sequence, Union

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

    def sample(
        self,
        dim_batch: Optional[Sequence[int]] = None,
        dim_expand: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        noise = self.sample_noise(
            dim_batch=dim_batch,
            dim_expand=dim_expand,
            dim_sample=self.size,
        )
        sigma = next(self.get_sigma())
        gamma = next(self.get_gamma())
        w = (noise + gamma) * sigma
        return w

    def forward(
        self,
        *dim: int,
    ) -> torch.Tensor:
        return self.sample(dim_batch=dim)

    def sample_pointwise(
        self,
        x: Union[int, float, torch.Tensor],
    ) -> torch.Tensor:
        if isinstance(x, (int, float)):
            dim_batch = None
            dim_expand = None
        elif isinstance(x, torch.Tensor):
            if len(x.shape) == 0:
                dim_batch = None
                dim_expand = None
            else:
                dim_batch_expand = x.shape[: -len(self.size)]
                if self.training:
                    dim_batch = dim_batch_expand
                    dim_expand = None
                else:
                    dim_batch = dim_batch_expand[:1]
                    dim_expand = dim_batch_expand[1:]
        else:
            raise NotImplementedError
        sample = self.sample(
            dim_batch=dim_batch,
            dim_expand=dim_expand,
        )
        return sample

    def __pos__(
        self,
    ) -> torch.Tensor:
        return self.sample()

    def __neg__(
        self,
    ) -> torch.Tensor:
        return -self.sample()

    def __add__(
        self,
        other: Union[int, float, torch.Tensor],
    ) -> torch.Tensor:
        sample = self.sample_pointwise(x=other)
        return sample + other

    def __radd__(
        self,
        other: Union[int, float, torch.Tensor],
    ) -> torch.Tensor:
        sample = self.sample_pointwise(x=other)
        return other + sample

    def __sub__(
        self,
        other: Union[int, float, torch.Tensor],
    ) -> torch.Tensor:
        sample = self.sample_pointwise(x=other)
        return sample - other

    def __rsub__(
        self,
        other: Union[int, float, torch.Tensor],
    ) -> torch.Tensor:
        sample = self.sample_pointwise(x=other)
        return other - sample

    def __mul__(
        self,
        other: Union[int, float, torch.Tensor],
    ) -> torch.Tensor:
        sample = self.sample_pointwise(x=other)
        return sample * other

    def __rmul__(
        self,
        other: Union[int, float, torch.Tensor],
    ) -> torch.Tensor:
        sample = self.sample_pointwise(x=other)
        return other * sample

    def __truediv__(
        self,
        other: Union[int, float, torch.Tensor],
    ) -> torch.Tensor:
        sample = self.sample_pointwise(x=other)
        return sample / other

    def __rtruediv__(
        self,
        other: Union[int, float, torch.Tensor],
    ) -> torch.Tensor:
        sample = self.sample_pointwise(x=other)
        return other / sample

    def __pow__(
        self,
        other: Union[int, float, torch.Tensor],
    ) -> torch.Tensor:
        sample = self.sample_pointwise(x=other)
        return sample**other

    def __rpow__(
        self,
        other: Union[int, float, torch.Tensor],
    ) -> torch.Tensor:
        sample = self.sample_pointwise(x=other)
        return other**sample

    def __rmatmul__(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        dim_batch_expand = x.shape[:-1]
        if self.training:
            dim_batch = dim_batch_expand
            dim_expand = None
        else:
            dim_batch = dim_batch_expand[:1]
            dim_expand = dim_batch_expand[1:]
        sigma = next(self.get_sigma())
        gamma = next(self.get_gamma())
        mu = gamma * sigma
        x_mu = x @ mu
        x_sigma = x @ sigma
        dim_sample = x_sigma.shape[-1:]
        noise = self.sample_noise(
            dim_batch=dim_batch,
            dim_expand=dim_expand,
            dim_sample=dim_sample,
        )
        y = x_sigma * noise + x_mu
        y = y.view(*dim_batch_expand, *dim_sample)
        return y
