from abc import abstractmethod  # ?? 抽象方法
from typing import Dict

from PIL import Image
import torch

from ...utils import ModuleBase


# 前面创立的一个更新模型参数的类


class TransformerBses(ModuleBase):  # 读取数据集

    default_hyper_params = dict()  # 很明显，这个参数继承了

    def __init__(self,  hps: Dict or None = None):
        super().__init__(hps)


    @abstractmethod
    def __call__(self, img:Image) -> Image or torch.tensor:  # 先把结构架好，后面，由子类去更改
        pass
