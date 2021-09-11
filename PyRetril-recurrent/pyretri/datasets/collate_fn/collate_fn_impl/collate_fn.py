from typing import Dict, List

import torch
from torch.utils.data.dataloader import default_collate

from ..collate_fn_base import CollateFnBase
from ...registry import COLLATEFS


@COLLATEFS.register
class CollateFn(CollateFnBase):
    default_hyper_params = dict()

    def __init__(self, hps: Dict or None = None):
        super().__init__(hps)

    def __call__(self, batch: List[Dict]) -> Dict[
        str, torch.tensor]:  # 输入的是每一个batch的数据，batch是一个list, # list 放了dict，dict应该是label,data...???
        assert isinstance(batch, list)
        assert isinstance(batch[0], dict)
        return default_collate(batch)  # 这里是什么操作？
