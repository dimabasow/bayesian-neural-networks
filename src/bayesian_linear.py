import torch
from .bayesian_module import BayesianModule
from .bayesian_block import BayesianBlock


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
        self.kl_weight = self.weight.kl
        if bias:
            self.bias = BayesianBlock(
                size=(out_features,),
                **factory_kwargs,
            )
            self.kl_bias = self.bias.kl
            self.forward = self.forward_bias_true
        else:
            self.register_parameter("bias", None)
            self.kl_bias = self.zero
            self.forward = self.forward_bias_false

    def reset_parameters(self) -> None:
        self.weight.reset_parameters()
        if self.bias is not None:
            self.bias.reset_parameters()

    def forward_bias_true(self, x: torch.Tensor) -> torch.Tensor:
        dim_extra = x.shape[:-1]
        noise = torch.normal(
            mean=0,
            std=1,
            size=(*dim_extra, self.out_features),
            dtype=self.dtype,
            device=self.device,
        )
        sigma = self.bias.sigma
        gamma = self.bias.gamma
        bias = sigma * gamma * noise
        y = self.forward_bias_false(x) + bias
        return y

    def forward_bias_false(self, x: torch.Tensor) -> torch.Tensor:
        dim_extra = x.shape[:-1]
        noise = torch.normal(
            mean=0,
            std=1,
            size=(*dim_extra, self.out_features),
            dtype=self.dtype,
            device=self.device,
        )
        sigma = self.weight.sigma
        gamma = self.weight.gamma
        mu = sigma * gamma
        y = (
            x@mu
            + (x@sigma)*noise
        )
        y = y.view(*dim_extra, self.out_features)
        return y

    @property
    def device(self) -> torch.types.Device:
        return self.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.weight.dtype

    @property
    def zero(self) -> torch.Tensor:
        return torch.zeros(
            size=(),
            device=self.device,
            dtype=self.dtype,
        )

    @property
    def kl(self) -> torch.Tensor:
        return self.kl_weight + self.kl_bias
