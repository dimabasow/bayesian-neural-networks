import torch
from .bayesian_module import BayesianModule
from .bayesian_block import BayesianBlock


class BayesianLinear(BayesianModule):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = BayesianBlock(
            size=(in_features, out_features),
            **factory_kwargs,
        )
        self.weight_kl = self.weight.kl
        if bias:
            self.bias = BayesianBlock(
                size=(out_features,),
                **factory_kwargs,
            )
            self.bias_kl = self.bias.kl
            self.forward = self.forward_bias_true
        else:
            self.register_parameter("bias", None)
            self.bias_kl = self.zero
            self.forward = self.forward_bias_false
    
    def reset_parameters(self) -> None:
        self.weight.reset_parameters()
        if self.bias is not None:
            self.bias.reset_parameters()
    
    def forward_bias_true(self, x: torch.Tensor) -> None:
        dim_extra = x.shape[:-1]
        x = x.view(-1, self.in_features)
        n = x.shape[0]
        noise = torch.normal(mean=0, std=1, size=(n, self.out_features))
        bias = self.bias(n).view(-1, self.out_features)
        y = (
            x@self.weight.mu
            + (x@self.weight.sigma)*noise
            + bias
        )
        y = y.view(*dim_extra, self.out_features)
        return y

    def forward_bias_false(self, x: torch.Tensor) -> None:
        dim_extra = x.shape[:-1]
        x = x.view(-1, self.in_features)
        n = x.shape[0]
        noise = torch.normal(mean=0, std=1, size=(n, self.out_features))
        y = (
            x@self.weight.mu
            + (x@self.weight.sigma)*noise
        )
        y = y.view(*dim_extra, self.out_features)
        return y

    @property
    def zero(self) -> torch.Tensor:
        return torch.zeros(
            size=(),
            device=self.weight.gamma.device,
            dtype=self.weight.gamma.dtype,
        )
    
    @property
    def kl(self) -> torch.Tensor:
        return self.weight_kl + self.bias_kl