from abc import abstractmethod  # ?? 抽象方法
from typing import Dict, List

import torch

from ...utils import ModuleBase  # 前面创立的一个更新模型参数的类


class CollateFnBase(ModuleBase):
    default_hyper_params = dict()  # 很明显，这个参数继承了

    def __init__(self, hps: Dict or None):
        super().__init__(hps)

    @abstractmethod
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.tensor]:  # 先把结构架好，后面，由子类去更改
        pass
