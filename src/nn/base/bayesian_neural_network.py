import copy
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import polars as pl
import torch

from src.nn.base.bayesian_module import BayesianModule


class BayesianNeuralNetwork(BayesianModule, ABC):
    @abstractmethod
    def __init__(self):
        super().__init__()
        self.__count_epoch = 0
        self.__metrics: Dict[int, Dict[str, float]] = {}

    @abstractmethod
    def configure_optimizer(
        self,
        optimizer: str,
        kwargs: Optional[Dict[str, Any]],
    ) -> torch.optim.Optimizer: ...

    @abstractmethod
    def negative_likelihood(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor: ...

    @property
    def current_epoch(self) -> int:
        return self.__count_epoch

    @property
    def last_metrics(self) -> Dict[str, float]:
        return copy.deepcopy(self.__metrics[self.current_epoch - 1])

    @property
    def df_metrics(self) -> pl.DataFrame:
        metrics = self.__metrics
        epochs = sorted(metrics.keys())
        values = [metrics[epoch] for epoch in epochs]
        df_metrics = pl.DataFrame(data=values)
        df_metrics = df_metrics.insert_column(
            index=0,
            column=pl.Series(name="index", values=epochs),
        )
        return df_metrics

    def log(self, key: str, value: float) -> None:
        epoch = self.current_epoch
        if epoch in self.__metrics:
            row = self.__metrics[epoch]
        else:
            row = {}
            self.__metrics[epoch] = row
        row[key] = value

    def loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        train_size: int,
    ) -> torch.Tensor:
        loss = self.negative_likelihood(x=x, y=y) + self.get_kl() / train_size
        return loss

    def init(
        self,
        x: torch.Tensor,
        optimizer: str = "Adam",
        lr: float = 0.1,
        num_epoch: int = 10,
    ):
        self.init_mode_on()
        optimizer: torch.optim.Optimizer = self.configure_optimizer(
            optimizer=optimizer, kwargs={"lr": lr}
        )
        for _ in range(num_epoch):
            self.train()
            optimizer.zero_grad()
            self(x)
            kl = self.get_kl()
            kl.backward()
            optimizer.step()
            self.log(key="kl", value=kl.item())
            self.__count_epoch += 1
        self.init_mode_off()

    def fit(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        optimizer: str = "Adam",
        lr: float = 0.01,
        num_epoch: int = 1000,
    ):
        optimizer: torch.optim.Optimizer = self.configure_optimizer(
            optimizer=optimizer, kwargs={"lr": lr}
        )
        for _ in range(num_epoch):
            optimizer.zero_grad()
            self.train()
            loss = self.loss(x=x, y=y, train_size=len(x))
            loss.backward()
            optimizer.step()
            self.log(key="loss", value=loss.item())
            self.log(key="p_item_average", value=torch.exp(-loss).item())
            self.__count_epoch += 1
