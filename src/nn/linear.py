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
        else:
            self.register_parameter("bias", None)
            self.forward = self.forward_bias_false

    def forward_bias_false(self, x: torch.Tensor) -> torch.Tensor:
        dim = x.shape[:-1]
        noise = torch.normal(
            mean=0,
            std=1,
            size=(*dim, self.out_features),
            dtype=self.dtype,
            device=self.device,
        )
        sigma = self.softplus(self.weight.rho)
        gamma = self.weight.gamma
        mu = sigma * gamma
        y = (
            x@mu
            + (x@sigma)*noise
        )
        y = y.view(*dim, self.out_features)
        return y

    def forward_bias_true(self, x: torch.Tensor) -> torch.Tensor:
        dim = x.shape[:-1]
        bias = self.bias(*dim)
        y = self.forward_bias_false(x) + bias
        return y
