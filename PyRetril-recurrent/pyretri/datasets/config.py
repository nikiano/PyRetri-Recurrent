from yacs.config import CfgNode

from .registry import COLLATEFS, FOLDERS, TRANSFORMERS
from ..utils import get_config_from_registry  #

"""
初始化设置cfg
这里的cfg里面没有内容，同时要注意cfg是两层的
cfg:CfgNode(CfgNode(dict(str,str)))
所以这里先设置get_XX函数，然后，用dataset把这些get输出再次包装到cfg里面去
"""

def get_collate_cfg() -> CfgNode:  # 初始化设置收集器
    cfg = get_config_from_registry(COLLATEFS)
    cfg["name"] = "unknown"
    return cfg


def get_folder_cfg():
    cfg = get_config_from_registry(FOLDERS)
    cfg["name"] = ["unknown"]
    return cfg


def get_transformers_cfg():
    cfg = get_config_from_registry(TRANSFORMERS)
    cfg["name"] = ["unknown"]
    return cfg


def get_datasets_cfg():
    cfg = CfgNode()
    cfg["collate_fn"] = get_collate_cfg()
    cfg["folder"] = get_folder_cfg()
    cfg["transformers"] = get_transformers_cfg()
    cfg["batch_size"] = 1
    return cfg
