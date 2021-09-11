import pickle  # 二进制版本的json
from abc import abstractmethod  # ?? 抽象方法
from typing import Dict, List

from PIL import Image

from ...utils import ModuleBase


# 前面创立的一个更新模型参数的类


class FolderBase(ModuleBase):  # 读取数据集
    default_hyper_params = dict()  # 很明显，这个参数继承了

    def __init__(self, data_json_path: str, transformer: callable or None = None, hps: Dict or None = None):
        super().__init__(hps)

        with open(data_json_path, "rb") as f:  # 解析pickle文件，这里面放的是特征
            self.data_info = pickle.load(f)

        self.data_json_path = data_json_path
        self.transformer = transformer

    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict:  # 先把结构架好，后面，由子类去更改
        pass

    def find_classes(self, info_dicts: Dict) -> (List, Dict):
        pass

    def read_img(self, path: str) -> Image: # 读数据
        try:
            img = Image.open(path)
            img = img.convert("RGB")
            return img
        except Exception as e:
            print('[DataSet]: WARNING image can not be loaded: {}'.format(str(e)))
            return None
