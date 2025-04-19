import random
from typing import Dict, Optional

import numpy as np
import polars as pl
import torch
from sklearn.metrics import roc_auc_score

from src.data.torch_tabular_dataset import TorchTabularDataset
from src.nn import BayesianBinaryPerceptrone


def polars_dict_mapper(df: pl.DataFrame, map_dict: Dict) -> pl.DataFrame:
    rules = []
    for column in map_dict:
        map_dict_col: Dict = map_dict[column]
        rule = None
        for key, value in map_dict_col.items():
            if rule is None:
                rule = pl.when(pl.col(column) == key).then(value)
            else:
                rule = rule.when(pl.col(column) == key).then(value)
        rule = rule.alias(column)
        rules.append(rule)
    df = df.with_columns(*rules)
    return df


def gini_score(
    y_true,
    y_score,
) -> np.float64:
    return (roc_auc_score(y_true=y_true, y_score=y_score) - 0.5) * 2


def inference_binary_perceptrone(
    model: BayesianBinaryPerceptrone,
    dataset: TorchTabularDataset,
    batch_size: Optional[int],
    sample_size: int,
) -> pl.DataFrame:
    name_target = dataset.metadata.targets_binary[0]
    result = {
        "score": [],
        "target": [],
    }
    model.eval()
    with torch.no_grad():
        for batch in dataset.to_bathes(
            batch_size=batch_size,
            shuffle=False,
            num_epochs=1,
        ):
            features = batch.features_numeric
            target = batch.target[name_target].value.int().tolist()
            score = (
                model(
                    (
                        features.view(1, *features.shape).expand(
                            sample_size, *features.shape
                        )
                    )
                )["score"]
                .mean(0)
                .view(-1)
                .tolist()
            )

            result["score"].extend(score)
            result["target"].extend(target)
    df_result = pl.DataFrame(result)
    return df_result


def make_experiment_binary_perceptrone(
    dataset_train: TorchTabularDataset,
    dataset_test: TorchTabularDataset,
    dim_hidden: int,
    batch_size_inference: Optional[int],
    sample_size_inference: int,
    random_seed: int,
):
    torch.random.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)

    name_target = dataset_train.metadata.targets_binary[0]
    model = BayesianBinaryPerceptrone(
        name_out="score",
        name_target=name_target,
        dim_in=len(dataset_train.metadata.features_numeric),
        dims_hidden=[dim_hidden],
    ).cuda()

    loss_init = model.init(
        features=dataset_train.data.features_numeric,
        num_epoch=1000,
    )

    model.train()
    loss_train = []
    for epoch_num, batch in enumerate(
        dataset_train.to_bathes(batch_size=None, shuffle=False, num_epochs=5_000)
    ):
        if epoch_num == 0:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0)
        elif epoch_num == 1_000:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
        elif epoch_num == 2_000:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0)
        optimizer.zero_grad()
        loss = model.loss(
            features=batch.features_numeric,
            target=batch.target,
            train_size=len(dataset_train),
        )
        loss.backward()
        optimizer.step()
        loss_train.append(loss.detach().item())

    df_inference_train = inference_binary_perceptrone(
        model=model,
        dataset=dataset_train,
        batch_size=batch_size_inference,
        sample_size=sample_size_inference,
    )

    df_inference_test = inference_binary_perceptrone(
        model=model,
        dataset=dataset_test,
        batch_size=batch_size_inference,
        sample_size=sample_size_inference,
    )

    return {
        "dim_hidden": dim_hidden,
        "gini_train": gini_score(
            y_true=df_inference_train["target"],
            y_score=df_inference_train["score"],
        ).item(),
        "gini_test": gini_score(
            y_true=df_inference_test["target"],
            y_score=df_inference_test["score"],
        ).item(),
        "loss_init": loss_init,
        "loss_train": loss_train,
    }
