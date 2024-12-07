from typing import Optional, Sequence
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
            "last_train_mean", torch.zeros(
                *size,
                **factory_kwargs,
                requires_grad=True,
            )
        )
        self.last_train_mean: Optional[Tensor]
        self.register_buffer(
            "running_std", torch.ones(
                *size,
                **factory_kwargs,
                requires_grad=True,
            )
        )
        self.running_std: Optional[Tensor]
        self.register_buffer(
            "last_train_std", torch.ones(
                *size,
                **factory_kwargs,
                requires_grad=True,
            )
        )
        self.last_train_std: Optional[Tensor]
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
        if self.training:
            self.num_batches_tracked.add_(1)
            if self.momentum is None:
                momentum = 1.0 / float(self.num_batches_tracked)
            else:
                momentum = self.momentum

        if self.momentum is None:
            momentum = 0.0
        else:
            momentum = self.momentum

        x_shape = x.shape
        x = x.view(-1, *self.size)
        if self.training:
            self.last_train_mean = x.mean(dim=0)
            self.last_train_std = x.std(dim=0)
            self.running_mean = (
                (self.running_mean.detach()*momentum)
                + (self.last_train_mean*(1 - momentum))
            )
            self.running_std = (
                (self.running_std.detach()*momentum)
                + (self.last_train_std*(1 - momentum))
            )

        if self.transform:
            x = (x - self.running_mean) / (self.running_std + self.eps)

        x = x.view(*x_shape)
        return x

    def get_kl(self) -> Tensor:
        if self.penalty:
            std_pow_2 = self.last_train_std ** 2
            mean_pow_2 = self.last_train_mean ** 2
            kl = (std_pow_2 + mean_pow_2 - torch.log(std_pow_2) - 1).sum() / 2
        else:
            kl = torch.zeros(
                size=[],
                dtype=self.dtype,
                device=self.device,
                requires_grad=True
            )
        return kl

    @property
    def device(self) -> torch.device:
        return self.running_mean.device

    @property
    def dtype(self) -> torch.dtype:
        return self.running_mean.dtype
