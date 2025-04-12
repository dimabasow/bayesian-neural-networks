import torch

from src.nn.base import BayesianModule


class BayesianSequential(BayesianModule, torch.nn.Sequential): ...


class BayesianModuleList(BayesianModule, torch.nn.ModuleList): ...


class BayesianModuleDict(BayesianModule, torch.nn.ModuleDict): ...
