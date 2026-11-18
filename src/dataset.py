"""Dataset：从 memmap 取图 + 增强。

只依赖 numpy/torch/torchvision（+ 可选 cv2 做 CLAHE），不引入 albumentations。

数据流：
    memmap[idx] (uint8 448×448)
      → 复制成可写数组
      → 可选 CLAHE（对比度受限直方图均衡，CXR 上能显著提升肋膈角等低对比结构）
      → torch 张量 (1,H,W) uint8
      → 几何/亮度增强（仅在训练时）
      → /255 → (x-mean)/std  归一化
      → 连同视位/体位/尺寸特征一起返回
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

# CLAHE 对象全局缓存一次：每次重建的代价不小，而它是无状态的
_CLAHE = None


def _get_clahe():
    global _CLAHE
    if _CLAHE is None:
        import cv2
        # clipLimit=2.0 是 CXR 上的常用值：足够拉开肺野对比，又不会放大噪声
        _CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return _CLAHE


def build_transforms(image_size: int, train: bool, level: str = "normal"):
    """构建增强流水线。单通道 (1,H,W) uint8 张量输入。

    刻意不用 albumentations：少一个国赛容器里的依赖，且 v2 能直接吃张量
    （包括 GPU 上的张量，未来可用 GPU 做增强进一步提速）。

    level="normal"（原方案）与 "strong" 的区别，都是为了缓解"模型在第 7 轮
    就过拟合训练集"这个实测现象：
      RandomResizedCrop   0.75~1.0 → 0.60~1.0（更强裁剪，逼模型看不完整的图）
      RandomRotation      ±10°     → ±15°
      ColorJitter         0.2      → 0.3
      RandomErasing       0.25     → 0.35

    各增强项的理由：
      RandomResizedCrop            模拟拍摄距离/裁剪差异，是收益最大的几何增强
      RandomHorizontalFlip          注意：X 光水平翻转会交换左右解剖位置。
                                    本赛题 10 个类别**没有**左右侧特异标签
                                    （气胸/胸腔积液不区分左右），所以安全，
                                    也是 CXR 领域标准增强。
      RandomRotation(±10°)          模拟患者体位轻微倾斜
      ColorJitter(brightness/contrast) 模拟不同 X 光机的曝光与窗宽差异
      RandomErasing                 随机遮挡，抑制模型依赖单一局部区域
    """
    from torchvision.transforms import v2

    if not train:
        # 验证/推理只做确定性的缩放
        return v2.Compose([v2.Resize((image_size, image_size), antialias=True)])

    if level == "weak":
        # 保留整幅胸廓；与 normal 的比较检验整套弱增强策略。
        return v2.Compose([
            v2.Resize((image_size, image_size), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomRotation(degrees=5, fill=0),
            v2.ColorJitter(brightness=0.1, contrast=0.1),
        ])
    if level == "strong":
        return v2.Compose([
            v2.RandomResizedCrop(image_size, scale=(0.60, 1.0), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomRotation(degrees=15, fill=0),
            v2.ColorJitter(brightness=0.3, contrast=0.3),
            v2.RandomErasing(p=0.35, inplace=False),
        ])
    if level == "normal":
        return v2.Compose([
            v2.RandomResizedCrop(image_size, scale=(0.75, 1.0), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomRotation(degrees=10, fill=0),      # fill=0 与 letterbox 黑边一致
            v2.ColorJitter(brightness=0.2, contrast=0.2),
            v2.RandomErasing(p=0.25, inplace=False),
        ])
    raise ValueError(f"未知增强强度 {level!r}，可选 normal/strong/weak")


class CXRMemmapDataset(Dataset):
    """基于 memmap 的数据集。

    Parameters
    ----------
    arr_path   : .npy memmap 路径（prepare_cache.py 的产物）
    rows       : index CSV 的子集 DataFrame，含 row_idx / view_idx / prop 等列
    image_size : 训练分辨率（如 384），与缓存分辨率（448）可以不同
    train      : True 启用增强
    mean/std   : 归一化统计量（来自 cache/norm.json）
    labels     : (N,10) uint8 多热标签；推理时传 None
    use_clahe  : 是否在线做 CLAHE。做成开关便于消融
    size_mean/std : 尺寸特征 z-score 的均值方差（来自 norm.json）
    cache_size : 缓存边长。给出时先缩放到 cache_size 再走增强，语义更清晰
    """

    def __init__(self, arr_path: str, rows, image_size: int, train: bool,
                 mean: float, std: float, labels: np.ndarray | None = None,
                 use_clahe: bool = True, size_mean=None, size_std=None,
                 cache_size: int | None = None, aug: str = "normal"):
        self.arr_path = arr_path
        self.image_size = image_size
        self.train = train
        self.mean = float(mean)
        self.std = float(std)
        self.labels = labels
        self.use_clahe = use_clahe
        self.aug = aug
        # 变换只构建一次（原来是每个 __getitem__ 都重新 Compose，纯属浪费）
        self._tf = build_transforms(image_size, train, aug)
        self.rows = rows.reset_index(drop=True) if hasattr(rows, "reset_index") else rows

        # 预先把所有列转成 numpy：__getitem__ 在训练热路径上，
        # 每个 epoch 会被调用 3 万次，用 pandas 的 .iloc[i] 逐行取值是明显的瓶颈。
        self.row_ids = np.asarray(self.rows["row_idx"], dtype=np.int64)
        # Study 在当前数据子集内的连续编号，用于训练时按检查组成 MIL batch。
        # 同一 Study 的图像在 fold 内总是属于同一划分，因此不会跨 train/val 泄漏。
        study_values = np.asarray(self.rows["Study_id"], dtype=str)
        _, self.study_idx = np.unique(study_values, return_inverse=True)
        self.study_idx = self.study_idx.astype(np.int64, copy=False)
        self.view_idx = np.asarray(self.rows["view_idx"], dtype=np.int64)
        self.orient_idx = np.asarray(self.rows["orient_idx"], dtype=np.int64)

        # 尺寸特征：log 变换后做 z-score。
        # log 是因为原图尺寸 2000~3000 分布右偏；它编码了设备/体位信息，
        # 与疾病相关但不该让模型直接记住"大图=某病"。
        log_h = np.log(np.clip(np.asarray(self.rows["Rows"], dtype=np.float64), 1, None))
        log_w = np.log(np.clip(np.asarray(self.rows["Columns"], dtype=np.float64), 1, None))
        sm = size_mean if size_mean is not None else [float(log_h.mean()), float(log_w.mean())]
        ss = size_std if size_std is not None else [float(log_h.std() or 1.0),
                                                   float(log_w.std() or 1.0)]
        self.size_feat = np.stack([
            (log_h - sm[0]) / ss[0],
            (log_w - sm[1]) / ss[1],
        ], axis=1).astype(np.float32)

        self._arr = None
        self._cache_size = cache_size

    # 惰性打开：DataLoader 的每个 worker 进程各自持有一份独立映射，
    # 不能把 np.memmap 对象直接 pickle 过去（会变成各自的内存副本，丧失共享）。
    @property
    def arr(self):
        if self._arr is None:
            self._arr = np.load(self.arr_path, mmap_mode="r")
        return self._arr

    def __len__(self) -> int:
        return len(self.rows)

    def _get_image(self, row_idx: int) -> np.ndarray:
        """取一张图（可选 CLAHE），返回可写的 uint8 数组。"""
        # np.array(...) 会复制：memmap 切片是只读视图，直接 torch.from_numpy
        # 会得到不可写张量，是未定义行为（torch 会告警）。
        # 单张 448² 仅 200KB，这次拷贝的代价可以忽略。
        img = np.array(self.arr[row_idx], dtype=np.uint8, copy=True)
        if self.use_clahe:
            img = _get_clahe().apply(img)
        return img

    def __getitem__(self, i: int) -> dict:
        img = self._get_image(int(self.row_ids[i]))
        tensor = torch.from_numpy(img).unsqueeze(0)  # (1,H,W) uint8，单通道

        # 缓存尺寸与目标尺寸不同时先缩放一次。
        # 训练时由 RandomResizedCrop 接管缩放，这里只在尺寸不匹配时兜底。
        size = self._cache_size or (
            max(tensor.shape[-2], tensor.shape[-1]) if not self.train else self.image_size
        )
        if size != tensor.shape[-1]:
            from torchvision.transforms import v2
            tensor = v2.functional.resize(tensor, [size, size], antialias=True)

        tensor = self._tf(tensor)
        # 归一化。用原地运算省一次内存分配（batch 64×1×384² 的拷贝不算小）
        tensor = tensor.to(torch.float32).div_(255.0)
        tensor = tensor.sub_(self.mean).div_(self.std)

        item = {
            "image": tensor,
            "view_idx": torch.tensor(self.view_idx[i], dtype=torch.long),
            "orient_idx": torch.tensor(self.orient_idx[i], dtype=torch.long),
            "size_feat": torch.from_numpy(self.size_feat[i]),
            # 全局行号（跨折唯一），用于把预测结果映射回 Study_id。
            # 注意这不是折内位置，训练脚本里索引要拿全量 meta/labels。
            "row_idx": torch.tensor(int(self.row_ids[i]), dtype=torch.long),
            "study_idx": torch.tensor(int(self.study_idx[i]), dtype=torch.long),
        }
        if self.labels is not None:
            item["target"] = torch.from_numpy(
                np.asarray(self.labels[i], dtype=np.float32))
        return item
