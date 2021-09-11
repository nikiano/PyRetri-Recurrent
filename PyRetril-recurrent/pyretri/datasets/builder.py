from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from yacs.config import CfgNode

from .collate_fn import CollateFnBase
from .folder import FolderBase
from .registry import COLLATEFS, FOLDERS, TRANSFORMERS
from ..utils import simple_build


def build_collate(cfg: CfgNode) -> CollateFnBase:
    """
    创建模型（实例化模型）
    """
    name = cfg["name"]
    collate = simple_build(name, cfg, COLLATEFS)  # 输出的是模型
    return collate


def build_transformers(cfg):
    """图像预处理"""
    names = cfg["name"]
    transformers = list()
    for name in names:
        transformers.append(simple_build(name, cfg, TRANSFORMERS))  # 一个list里面放上所有初始化格式图像预处理的模型（函数）
    transformers = Compose(transformers)  # torchvision 的Compose里面是list !!
    return transformers


def build_folder(data_json_path: str, cfg: CfgNode) -> FolderBase:  # ??? 这里非常不懂，为什么输出是folderbase??
    """
    生成数据集
    """
    trans = build_transformers(cfg.transformers)
    folder = simple_build(cfg.folder["name"], cfg.folder, FOLDERS, data_json_path=data_json_path,
                          transformer=trans)  # 这里的关键字参数不知做了什么作用，需要debug,inset
    return folder


def build_loader(folder: FolderBase, cfg: CfgNode) -> DataLoader:  # 有预感这里的folder将来自于build_folder函数
    co_fn = build_collate(
        cfg.collate_fn
    )  # 这个是在读文件yaml时传入的参数名字，name of colloate_fn（是datasets参数中的collate_fn值。 ==>默认的是CollateFn
    data_loader = DataLoader(folder, cfg["batch_size"], collate_fn=co_fn, num_workers=8,
                             pin_memory=True)  # 这里报错期待的是dataset，输入的是folderbase,folderbase是生成dataset的
    return data_loader
