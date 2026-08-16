import json
import os
import random
from typing import Dict, List, Optional

import numpy as np
import polars as pl
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.data.torch_tabular_dataset import TorchTabularDataset
from src.data.preprocessing.preprocessor import RuleTransform, Preprocessor
from src.nn import BayesianBinaryPerceptrone, BayesianNeuralNetwork, BinaryPerceptrone


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
    random_seed: int = 0,
) -> pl.DataFrame:
    torch.random.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)

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
    torch.cuda.empty_cache()
    df_result = pl.DataFrame(result)
    return df_result


def train_bayesian_model(
    dataset: TorchTabularDataset,
    model: BayesianNeuralNetwork,
    optimizer: str = "Adam",
    lr: float = 0.1,
    num_epoch: int = 1000,
    batch_size: int | None = None,
) -> List[float]:
    optimizer: torch.optim.Optimizer = getattr(torch.optim, optimizer)(
        model.parameters(),
        lr=lr,
        weight_decay=0,
    )
    model.train()
    loss_train = []
    for _, batch in enumerate(
        dataset.to_bathes(batch_size=batch_size, shuffle=True, num_epochs=num_epoch)
    ):
        optimizer.zero_grad()
        loss = model.loss(
            features=batch.features_numeric,
            target=batch.target,
            train_size=len(dataset),
        )
        loss.backward()
        optimizer.step()
        loss_train.append(loss.detach().item())
    torch.cuda.empty_cache()
    return loss_train


def make_experiment_bayesian_binary_perceptrone(
    dataset_train: TorchTabularDataset,
    dataset_test: TorchTabularDataset,
    dim_hidden: int,
    n_hidden: int,
    f_act: "str",
    batch_size_inference: Optional[int],
    sample_size_inference: int,
    random_seed: int,
    log_loss_init: bool = False,
    log_loss_train: bool = False,
) -> dict[str, float]:
    torch.random.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)

    name_target = dataset_train.metadata.targets_binary[0]
    model = BayesianBinaryPerceptrone(
        name_out="score",
        name_target=name_target,
        dim_in=len(dataset_train.metadata.features_numeric),
        dims_hidden=[dim_hidden] * n_hidden,
        f_act=f_act,
        batch_norm=False,
        batch_affine=False,
        batch_penalty=False,
    ).cuda()

    loss_init = model.init(
        features=dataset_train.data.features_numeric,
        optimizer="Adam",
        num_epoch=100,
        lr=0.00001,
    )
    loss_init.extend(
        model.init(
            features=dataset_train.data.features_numeric,
            optimizer="Adam",
            num_epoch=500,
            lr=0.001,
        )
    )

    loss_train = train_bayesian_model(
        dataset=dataset_train,
        model=model,
        optimizer="Adam",
        lr=0.1,
        num_epoch=3000,
    )
    loss_train.extend(
        train_bayesian_model(
            dataset=dataset_train,
            model=model,
            optimizer="Adam",
            lr=0.01,
            num_epoch=1000,
        )
    )
    loss_train.extend(
        train_bayesian_model(
            dataset=dataset_train,
            model=model,
            optimizer="Adam",
            lr=0.001,
            num_epoch=1000,
        )
    )
    loss_train.extend(
        train_bayesian_model(
            dataset=dataset_train,
            model=model,
            optimizer="Adam",
            lr=0.0001,
            num_epoch=1000,
        )
    )
    loss_train.extend(
        train_bayesian_model(
            dataset=dataset_train,
            model=model,
            optimizer="Adam",
            lr=0.00001,
            num_epoch=1000,
        )
    )

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
    result = {
        "roc_auc_train": roc_auc_score(
            y_true=df_inference_train["target"],
            y_score=df_inference_train["score"],
        ).item(),
        "roc_auc_test": roc_auc_score(
            y_true=df_inference_test["target"],
            y_score=df_inference_test["score"],
        ).item(),
    }
    if log_loss_init:
        result["loss_init"] = loss_init
    if log_loss_train:
        result["loss_train"] = loss_train

    return result


def train_classical_model(
    dataset: TorchTabularDataset,
    model: BayesianNeuralNetwork,
    weight_decay: float = 0,
    batch_size: int | None = None,
) -> List[float]:
    model.train()
    loss_train = []
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01,
        weight_decay=weight_decay,
    )
    for batch in dataset.to_bathes(batch_size=batch_size, shuffle=True, num_epochs=5_000):
        optimizer.zero_grad()
        loss = model.loss(
            features=batch.features_numeric,
            target=batch.target,
            train_size=len(dataset),
        )
        loss.backward()
        optimizer.step()
        loss_train.append(loss.detach().item())
    torch.cuda.empty_cache()
    return loss_train


def make_experiment_classical_binary_perceptrone(
    dataset_train: TorchTabularDataset,
    dataset_test: TorchTabularDataset,
    dim_hidden: int,
    n_hidden: int,
    f_act: "str",
    weight_decay: float,
    random_seed: int,
    log_loss_train: bool = False,
):
    torch.random.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)

    name_target = dataset_train.metadata.targets_binary[0]
    model = BinaryPerceptrone(
        name_out="score",
        name_target=name_target,
        dim_in=len(dataset_train.metadata.features_numeric),
        dims_hidden=[dim_hidden] * n_hidden,
        f_act=f_act,
    ).cuda()

    loss_train = train_classical_model(
        dataset=dataset_train,
        model=model,
        weight_decay=weight_decay,
    )

    df_inference_train = inference_binary_perceptrone(
        model=model,
        dataset=dataset_train,
        batch_size=None,
        sample_size=1,
    )

    df_inference_test = inference_binary_perceptrone(
        model=model,
        dataset=dataset_test,
        batch_size=None,
        sample_size=1,
    )
    result = {
        "roc_auc_train": roc_auc_score(
            y_true=df_inference_train["target"],
            y_score=df_inference_train["score"],
        ).item(),
        "roc_auc_test": roc_auc_score(
            y_true=df_inference_test["target"],
            y_score=df_inference_test["score"],
        ).item(),
    }
    if log_loss_train:
        result["loss_train"] = loss_train

    return result


def make_experiment_binary_perceptrone(
    dataset_train: TorchTabularDataset,
    dataset_test: TorchTabularDataset,
    dim_hidden: int,
    n_hidden: int,
    f_act: "str",
    batch_size_inference_bayesian: Optional[int],
    sample_size_inference_bayesian: int,
    random_seed: int,
    log_loss_init: bool = False,
    log_loss_train: bool = False,
) -> dict[str, dict[str, float]]:
    result = {
        "random_seed": random_seed,
        "n_hidden": n_hidden,
        "dim_hidden": dim_hidden,
        "f_act": f_act,
    }
    result["bayesian"] = make_experiment_bayesian_binary_perceptrone(
        dataset_train=dataset_train,
        dataset_test=dataset_test,
        dim_hidden=dim_hidden,
        n_hidden=n_hidden,
        f_act=f_act,
        batch_size_inference=batch_size_inference_bayesian,
        sample_size_inference=sample_size_inference_bayesian,
        random_seed=random_seed,
        log_loss_init=log_loss_init,
        log_loss_train=log_loss_train,
    )
    result["classic"] = (
        make_experiment_classical_binary_perceptrone(
            dataset_train=dataset_train,
            dataset_test=dataset_test,
            dim_hidden=dim_hidden,
            n_hidden=n_hidden,
            f_act=f_act,
            weight_decay=0,
            random_seed=random_seed,
            log_loss_train=log_loss_train,
        )
    )
    return result


def make_experiments_binary_perceptrone(
    path_to_save: str,
    df: pl.DataFrame,
    rules_tranform: list[RuleTransform],
    n_hidden_layers: list[int],
    dims_hidden: list[int],
    random_seeds: list[int],
    f_act: str,
    batch_size_inference_bayesian: Optional[int],
    sample_size_inference_bayesian: int,
):
    if not os.path.exists(path=path_to_save):
        os.mkdir(path=path_to_save)
    for random_seed in random_seeds:
        df_train, df_test = train_test_split(df, test_size=0.2, random_state=random_seed)
        preprocessor = Preprocessor.from_rules(*rules_tranform)
        df_train_transformed = preprocessor.fit_transform(data=df_train)
        df_test_transformed = preprocessor.transform(data=df_test)

        dataset_train = TorchTabularDataset(
            df=df_train_transformed,
            metadata=preprocessor.metadata,
        ).cuda()
        dataset_test = TorchTabularDataset(
            df=df_test_transformed,
            metadata=preprocessor.metadata,
        ).cuda()
        results = []
        for dim_hidden in dims_hidden:
            for n_hidden in n_hidden_layers:
                path_output = os.path.join(
                    path_to_save, f"n_hidden_{n_hidden}_dim_hidden_{dim_hidden}_f_act_{f_act}.json"
                )
                if not os.path.exists(path=path_output):
                    result = make_experiment_binary_perceptrone(
                        dataset_train=dataset_train,
                        dataset_test=dataset_test,
                        dim_hidden=dim_hidden,
                        n_hidden=n_hidden,
                        f_act=f_act,
                        batch_size_inference_bayesian=batch_size_inference_bayesian,
                        sample_size_inference_bayesian=sample_size_inference_bayesian,
                        random_seed=random_seed,
                    )
                    with open(path_output, "w") as f:
                        json.dump(result, f, indent=4)
                else:
                    with open(path_output, "r") as f:
                        result = json.load(f)
                results.append(result)
    return results


# def make_experiments_binary_perceptrone(
#     path_to_save: str,
#     dims_hidden: List[int],
#     dataset_train: TorchTabularDataset,
#     dataset_test: TorchTabularDataset,
#     n_hidden: int,
#     weight_decays_classic: List[float],
#     batch_size_inference_bayesian: Optional[int],
#     sample_size_inference_bayesian: int,
#     random_seed: int,
# ):
#     if not os.path.exists(path=path_to_save):
#         os.mkdir(path=path_to_save)
#     data = []
#     for dim_hidden in dims_hidden:
#         path_output = os.path.join(
#             path_to_save, f"n_hidden_{n_hidden}_dim_hidden_{dim_hidden}.json"
#         )
#         if not os.path.exists(path=path_output):
#             result = make_experiment_binary_perceptrone(
#                 dataset_train=dataset_train,
#                 dataset_test=dataset_test,
#                 dim_hidden=dim_hidden,
#                 n_hidden=n_hidden,
#                 weight_decays_classic=weight_decays_classic,
#                 batch_size_inference_bayesian=batch_size_inference_bayesian,
#                 sample_size_inference_bayesian=sample_size_inference_bayesian,
#                 random_seed=random_seed,
#             )
#             with open(path_output, "w") as f:
#                 json.dump(result, f, indent=4)
#         else:
#             with open(path_output, "r") as f:
#                 result = json.load(f)
#                 for experiment in tuple(result.keys()):
#                     for metric in tuple(result[experiment].keys()):
#                         result[f"{experiment}_{metric}"] = result[experiment][metric]
#                     result.pop(experiment)
#                 result["n_hidden"] = n_hidden
#                 result["dim_hidden"] = dim_hidden
#                 data.append(result)
#     df = pl.DataFrame(data).sort("dim_hidden")
#     df = df[
#         ["n_hidden", "dim_hidden"]
#         + [column for column in df.columns if column not in ["n_hidden", "dim_hidden"]]
#     ]
#     df.write_csv(
#         os.path.join(
#             path_to_save,
#             f"n_hidden_{n_hidden}_dims_hidden_{dims_hidden[0]}_{dims_hidden[-1]}.csv",
#         )
#     )
