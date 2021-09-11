from typing import Dict

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import CenterCrop as TCenterCrop
from torchvision.transforms import Resize as TResize
from torchvision.transforms import TenCrop as TTenCrop
from torchvision.transforms import ToTensor as TToTensor
from torchvision.transforms.functional import hflip  # 将指定图像水平翻折

from ..transformers_base import TransformerBses
from ...registry import TRANSFORMERS

"""
奶奶的重写的transforms的以上这些类，有点类，周一再继续改把。回家买电脑~~~~期待新电脑
"""


@TRANSFORMERS.register
class DirectResize(TransformerBses):
    """这里新建类的目的就是调用transforms函数？
    或许想统一在注册表中的格式，都是，key=类名，value=类
    这样？因为后面基本
    """
    default_hyper_params = {
        "size": (224, 224),
        "interpolation": Image.BILINEAR
    }

    def __init__(self, hps: Dict or None = None):
        super().__init__(hps)
        self.t_transformer = TResize(self._hyper_params["size"], self._hyper_params["interpolation"])

    def __call__(self, img: Image) -> Image:
        return self.t_transformer(img)


@TRANSFORMERS.register
class PadResize(TransformerBses):
    default_hyper_params = {
        "size": 224,
        "padding_v": [124, 116, 104],
        "interpolation": Image.BILINEAR
    }

    def __init__(self, hps: Dict or None = None):
        super().__init__(hps)

    def __call__(self, img: Image) -> Image:
        """
        判断
        """
        target_size = self._hyper_params["size"]
        padding_v = tuple(self._hyper_params["padding_v"])
        interpolation = self._hyper_params["interpolation"]

        w, h = img.resize
        if w > h:  # 保持原有比例，缩小或放到到t。
            img = img.resize((int(target_size), int(h * target_size * 1.0 / w)), interpolation)
        else:
            img = img.resize((int(w * target_size * 1.0 / h), int(target_size)), interpolation)

        ret_img = Image.new("RGB", (target_size, target_size), padding_v)  # 根据提供的格式和大小，创建一个新的图片，
        w, h = img.size
        st_w = int((ret_img.size[0] - w) / 2.0)
        st_h = int((ret_img.size[1] - w) / 2.0)
        ret_img.paste(img, (st_w, st_h))  # 将内容粘贴到图片上去 ？？？？
        return ret_img


@TRANSFORMERS.register
class ShorterResize(TransformerBses):
    default_hyper_params = {
        "size": 224,
        "interpolation": Image.BILINEAR
    }

    def __init__(self, hps: Dict or None = None):
        super().__init__(hps)
        self.t_transformer = TResize(self._hyper_param["size"], self._hyper_params["interpolation"])  # H,W

    def __call__(self, img: Image) -> Image:
        return self.t_transformer(img)


@TRANSFORMERS.register
class CenterCrop(TransformerBses):
    default_hyper_params = {
        "size": 224,
    }

    def __init__(self, hps: Dict or None = None):
        super().__init__(hps)
        self.t_transformer = TCenterCrop(self._hyper_param["size"], )

    def __call__(self, img: Image) -> Image:
        return self.t_transformer(img)


@TRANSFORMERS.register
class ToTensor(TransformerBses):
    default_hyper_params = dict()

    def __init__(self, hps):
        super().__init__(hps)
        self.t_transformer = TToTensor()

    def __call__(self, imgs: Image or tuple) -> torch.Tensor:
        if not isinstance(imgs, tuple):
            imgs = [imgs]
        ret_tensor = list()
        for img in imgs:
            ret_tensor.append(self.t_transformer(img))  # 感觉像是把一个batch的图片全部拼接起来。
        ret_tensor = torch.stack(ret_tensor, dim=0)  # （pytorch 的拼接函数）
        return ret_tensor


@TRANSFORMERS.register
class ToCaffeTensor(TransformerBses):
    """
    官方解释：为再caffe中训练的模型创建张量
    """
    default_hyper_params = dict()

    def __init__(self, hps):
        super().__init__(hps)

    def __call__(self, imgs: Image or tuple) -> torch.tensor:
        if not isinstance(imgs, tuple):
            imgs = [imgs]

        ret_tensor = list()
        for img in imgs:
            img = np.array(img, np.int32, copy=False)  # 变为向量处理
            r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
            img = np.stack([b, g, r], axis=2)  # 应该也是一种拼接，难受。先放着，后面debug再看（就是堆叠，按维度）
            # （https://numpy.org/doc/stable/reference/generated/numpy.vstack.html）
            img = torch.from_numpy(img)
            img = img.transpose(0, 1).transpose(0,
                                                2).contiguous()  # transpose:使输入的两个维度转换。(swapped),contiguous：用于保证tensor是contigous（连续）的
            # https://zhuanlan.zhihu.com/p/64551412
            img = img.float()
            ret_tensor.append(img)
        ret_tensor = torch.stack(ret_tensor, dim=0)
        return ret_tensor


@TRANSFORMERS.register
class Normalize(TransformerBses):
    default_hyper_params = {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    }

    def __init__(self, hps: Dict or None = None):
        super().__init__(hps)
        for v in ["mean", "std"]:  # 这里的操作不是很懂，先累积把
            self.__dict__[v] = np.array(self._hpyper_params[v])[None, :, None, None]
            self.__dict__[v] = torch.from_numpy(self.__dict__[v]).float()

    def __call__(self, tensor: torch.tensor) -> torch.tensor:
        assert tensor.ndimension() == 4
        tensor.sub_(self.mean).div_(self.std)  # 均值化， 这里的计算很有趣，应该又的是内部函数，直接改原值
        return tensor


@TRANSFORMERS.register
class TenCrop(TransformerBses):
    default_hyper_params = {
        "size": 224
    }

    def __init__(self, hps):
        super().__init__(hps)
        self.t_transformer = TTenCrop(self._hyper_parmas["size"])

    def __call__(self,
                 img: Image) -> Image:  # 类实例可以做函数用，（http://c.biancheng.net/view/2380.html）a = A(),a(x)==>这个x就会放到call函数中
        return self.t_transformer(img)


@TRANSFORMERS.register
class TwoFlip(TransformerBses):
    default_hyper_params = dict()

    def __init__(self, hps):
        super().__init__(hps)

    def __call__(self, img: Image) -> (Image, Image):
        return img, hflip(img)  # 水平翻折
