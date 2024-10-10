import torch
from src.nn.base import BayesianModule, BayesianBlock


class BayesianLinear(BayesianModule):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: torch.types.Device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = BayesianBlock(
            size=(in_features, out_features),
            **factory_kwargs,
        )
        if bias:
            self.bias = BayesianBlock(
                size=(out_features,),
                **factory_kwargs,
            )
            self.forward = self.forward_bias_true
            self.get_kl_proxy = self.get_kl_bias_true
        else:
            self.register_parameter("bias", None)
            self.forward = self.forward_bias_false
            self.get_kl_proxy = self.get_kl_bias_false

    def forward_bias_false(self, x: torch.Tensor) -> torch.Tensor:
        dim_extra = x.shape[:-1]
        noise = torch.normal(
            mean=0,
            std=1,
            size=(*dim_extra, self.out_features),
            dtype=self.dtype,
            device=self.device,
        )
        sigma = self.weight.get_sigma()
        gamma = self.weight.gamma
        mu = sigma * gamma
        y = (
            x@mu
            + (x@sigma)*noise
        )
        y = y.view(*dim_extra, self.out_features)
        return y

    def get_kl_bias_false(self) -> torch.Tensor:
        return self.weight.get_kl()

    def forward_bias_true(self, x: torch.Tensor) -> torch.Tensor:
        dim_extra = x.shape[:-1]
        noise = torch.normal(
            mean=0,
            std=1,
            size=(*dim_extra, self.out_features),
            dtype=self.dtype,
            device=self.device,
        )
        sigma = self.bias.get_sigma()
        gamma = self.bias.gamma
        bias = sigma * gamma * noise
        y = self.forward_bias_false(x) + bias
        return y

    def get_kl_bias_true(self) -> torch.Tensor:
        return self.get_kl_bias_false() + self.bias.get_kl()

    def get_kl(self) -> torch.Tensor:
        return self.get_kl_proxy()
