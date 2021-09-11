from typing import Dict, List

from ..folder_base import FolderBase
from ...registry import FOLDERS


@FOLDERS.register
class Folder(FolderBase):
    default_hyper_params = {"use_bbox": False}

    def __init__(self, data_json_path: str, transformer: callable or None = None, hps: Dict or None = None):
        super().__init__(data_json_path, transformer, hps)
        self.clases, self.class_to_idx = self.find_classes(self.data_info["info_dicts"])

    def find_classes(self, info_dicts: Dict) -> (List, Dict):
        """
        提取数据的类别(list，及，每个类下面对应的数据(dict)（有可能随数据的Index,+数据）
        info_dicts应该是从pictle二进制特征文件里面解析出来的东西

        Get the class names and the mapping relations.

        """
        classes = list()
        for i in range(len(info_dicts)):
            if info_dicts[i]["label"] not in classes:
                classes.append(info_dicts[i]["label"])
        classes.sort()
        class_to_idx = {classes[i] for i in range(len(classes))}
        return classes, class_to_idx

    def __len__(self) -> int:
        return len(self.data_info["info_dicts"])

    def __getitem__(self, idx: int) -> Dict:
        """
        获取数据：原图array，pickle解压的数据，数据所属的类

        天啊，这里又box数据耶，我不需要他画出box里面的内容，我只需要判断整张图片属于哪类，
        也就是说，我的图片只属于一个类，而不是图片里的某小部分各自属于哪个类

        """
        info = self.data_info["info_dicts"][idx]  # 从pickle里面解析的数据特征
        img = self.read_img(info["path"])   # 原图array

        if self._hyper_params["use_bbox"]: # 这里是超参的值，从 Moudul_Base上面拿的   还是涉及到default_hyper_params，不知道怎么回事
            assert info["bbox"] is not None, "image {} does not have a bbox".format(info["path"])
            x1, y1, x2, y2 = info["bbox"]
            box = map(int, (x1, y1, x2, y2))
            img = img.crop(box) # 还好这里用了裁剪
        img = self.transformer(img) # 然后在预处理下裁剪的内容
        return {"img": img, "idx": idx, "label": self.class_to_idx[info["label"]]} # idx是pickle里解压的数据，img是图片按box裁剪后的array
