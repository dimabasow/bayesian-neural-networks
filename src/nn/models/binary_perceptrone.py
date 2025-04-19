from typing import Any, Dict, Literal, Optional, Sequence, Union

from src.nn.backbones.perceptrone import Perceptrone
from src.nn.base.bayesian_neural_network import BayesianNeuralNetwork
from src.nn.container import BayesianModuleList
from src.nn.heads.binary_classification_head import BinaryClassificationHead


class BinaryPerceptrone(BayesianNeuralNetwork):
    def __init__(
        self,
        name_out: str,
        name_target: str,
        dim_in: int,
        dims_hidden: Sequence[int],
        f_act: Union[
            Literal["ELU"],
            Literal["ReLU"],
            Literal["LeakyReLU"],
        ] = "LeakyReLU",
        f_act_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.backbones = BayesianModuleList(
            [
                Perceptrone(
                    name_in=None,
                    name_out="embeding",
                    dim_in=dim_in,
                    dim_out=1,
                    dims_hidden=dims_hidden,
                    f_act=f_act,
                    f_act_kwargs=f_act_kwargs,
                )
            ]
        )
        self.heads = BayesianModuleList(
            [
                BinaryClassificationHead(
                    name_in="embeding",
                    name_out=name_out,
                    name_target=name_target,
                )
            ]
        )
