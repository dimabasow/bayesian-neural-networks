from typing import Optional
from collections.abc import Sequence
import torch
from torch import Tensor
from src.nn.base import BayesianModule


class BayesianBatchNorm(BayesianModule):
    __constants__ = [
        "size",
        "transform",
        "penalty",
        "momentum",
        "eps",
    ]
    size: Sequence[int]
    transform: bool
    penalty: bool
    momentum: Optional[float]
    eps: float

    def __init__(
        self,
        size: Sequence[int],
        transform: bool = True,
        penalty: bool = True,
        momentum: Optional[float] = 0.1,
        eps: float = 1e-5,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.size = size
        self.transform = transform
        self.penalty = penalty
        self.momentum = momentum
        self.eps = eps
        self.register_buffer(
            "running_mean", torch.zeros(
                *size,
                **factory_kwargs,
                requires_grad=True,
            )
        )
        self.running_mean: Optional[Tensor]
        self.register_buffer(
            "running_std", torch.ones(
                *size,
                **factory_kwargs,
                requires_grad=True,
            )
        )
        self.running_std: Optional[Tensor]
        self.register_buffer(
            "num_batches_tracked",
            torch.tensor(
                0,
                dtype=torch.long,
                **{k: v for k, v in factory_kwargs.items() if k != "dtype"},
            ),
        )
        self.num_batches_tracked: Optional[Tensor]
        self.register_buffer(
            "kl", torch.zeros([], **factory_kwargs, requires_grad=True)
        )
        self.kl: Optional[Tensor]
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.running_mean.detach_().zero_().requires_grad_(True)
        self.running_std.detach_().fill_(1).requires_grad_(True)
        self.kl.detach_().zero_().requires_grad_(True)
        self.num_batches_tracked.zero_()

    def forward(self, x: Tensor) -> Tensor:
        if self.momentum is None:
            momentum = 0.0
        else:
            momentum = self.momentum

        if self.training:
            bn_training = True
            self.num_batches_tracked.add_(1)
            if self.momentum is None:
                momentum = 1.0 / float(self.num_batches_tracked)
            else:
                momentum = self.momentum

        else:
            bn_training = (
                (self.running_mean is None) and (self.running_std is None)
            )

        x_shape = x.shape
        x = x.view(-1, *self.size)
        if bn_training:
            mean = x.mean(dim=0)
            std = x.std(dim=0)
            self.running_mean = (
                (self.running_mean.detach()*momentum)
                + (mean*(1 - momentum))
            )
            self.running_std = (
                (self.running_std.detach()*momentum)
                + (std*(1 - momentum))
            )
        if self.transform:
            x = (x - self.running_mean) / (self.running_std + self.eps)
        if self.penalty:
            std_pow_2 = self.running_std ** 2
            mu_pow_2 = self.running_mean ** 2
            self.kl = (std_pow_2 + mu_pow_2 - torch.log(std_pow_2) - 1).sum() / 2
        else:
            self.kl = torch.zeros(
                size=[],
                dtype=self.dtype,
                device=self.device,
                requires_grad=True
            )
        x = x.view(*x_shape)
        return x

    def get_kl(self) -> Tensor:
        return self.kl

    @property
    def device(self) -> torch.device:
        return self.running_mean.device

    @property
    def dtype(self) -> torch.dtype:
        return self.running_mean.dtype
