from abc import abstractmethod, ABC
import torch
import copy
import pandas as pd
from src.nn.base import BayesianModule


class BayesianNeuralNetwork(BayesianModule, ABC):

    @abstractmethod
    def __init__(self):
        super().__init__()
        self.__count_epoch = 0
        self.__metrics: dict[0, dict[str, float]] = {}

    @abstractmethod
    def configure_optimizer(self) -> torch.optim.Optimizer:
        ...

    @abstractmethod
    def negative_likelihood(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        ...

    @property
    def current_epoch(self) -> int:
        return self.__count_epoch

    @property
    def last_metrics(self) -> dict[str, float]:
        return copy.deepcopy(self.__metrics[self.current_epoch - 1])

    @property
    def df_metrics(self) -> pd.DataFrame:
        metrics = self.__metrics
        epochs = sorted(metrics.keys())
        values = [metrics[epoch] for epoch in epochs]
        df_metrics = pd.DataFrame(data=values)
        df_metrics["epoch"] = epochs
        df_metrics = df_metrics.set_index("epoch")
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

    def fit(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        num_epoch: int = 1000,
    ):
        optimizer = self.configure_optimizer()

        for _ in range(num_epoch):
            optimizer.zero_grad()
            self.train()
            loss = self.loss(x=x, y=y, train_size=len(x))
            loss.backward()
            optimizer.step()
            self.log(key="loss", value=loss.item())
            self.log(key="p_item_average", value=torch.exp(-loss).item())
            self.__count_epoch += 1