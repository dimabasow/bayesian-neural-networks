from typing import Optional
import torch
from src.nn.base import BayesianModule, BayesianParameter


class BayesianLinear(BayesianModule):

    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[torch.types.Device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.register_buffer(
            "scale", 1 / torch.sqrt(torch.tensor(self.in_features))
        )
        self.scale: torch.Tensor
        self.out_features = out_features
        self.weight = BayesianParameter(
            size=(in_features, out_features),
            **factory_kwargs,
        )
        if bias:
            self.bias = BayesianParameter(
                size=(out_features,),
                **factory_kwargs,
            )
            self.forward = self.forward_bias_true
        else:
            self.register_module("bias", None)
            self.forward = self.forward_bias_false

    def forward_bias_false(self, x: torch.Tensor) -> torch.Tensor:
        y = (x @ self.weight) * self.scale
        if self.is_init_mode_on:
            self.last_mean = y.view(-1, y.shape[-1]).mean(dim=0)
            self.last_std = y.view(-1, y.shape[-1]).std(dim=0)
        return y

    def forward_bias_true(self, x: torch.Tensor) -> torch.Tensor:
        y = self.forward_bias_false(x) + self.bias
        return y

    def get_kl(self):
        kl = super().get_kl()
        if self.is_init_mode_on:
            std_pow_2 = self.last_std ** 2
            mean_pow_2 = self.last_mean ** 2
            kl_z = (
                (std_pow_2 + mean_pow_2 - torch.log(std_pow_2) - 1).sum() / 2
            )
            kl = kl + kl_z

        return kl
