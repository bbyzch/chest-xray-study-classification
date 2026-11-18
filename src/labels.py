"""标签解析、索引构建、视位/体位词表。

赛题 10 个类别（id 1..10）：
    1 纵隔心影增大   2 气胸   3 肺部实变   4 肺炎   5 肺水肿
    6 心影扩大       7 肺不张 8 肺部阴影   9 胸腔积液 10 无异常

约定：内部一律使用 **0-based 索引 0..9**，即类别 id 1 → 索引 0，类别 id 10 → 索引 9。
提交文件里要还原成 1-based 的 id（见 submit.py）。

本模块是纯 pandas/numpy，不依赖 torch，可以脱离 GPU 完整测试。
"""
from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import pandas as pd

NUM_CLASSES = 10

CLASS_IDS = list(range(1, NUM_CLASSES + 1))

# 类别名仅用于报告和日志，顺序必须与索引 0..9 严格对应
CLASS_NAMES_ZH = [
    "纵隔心影增大", "气胸", "肺部实变", "肺炎", "肺水肿",
    "心影扩大", "肺不张", "肺部阴影", "胸腔积液", "无异常",
]
CLASS_NAMES_EN = [
    "Enlarged Cardiomediastinum", "Pneumothorax", "Consolidation", "Pneumonia", "Edema",
    "Cardiomegaly", "Atelectasis", "Lung Opacity", "Pleural Effusion", "No Finding",
]

# 实测出现过的取值：AP / PA / LATERAL / LL / "PA LLD" / 缺失
# 词表顺序即 embedding 的下标，**一旦训练过就不能再改**，否则权重与索引错位
VIEW_VOCAB = ["AP", "PA", "LATERAL", "LL", "LLD", "UNKNOWN"]
ORIENT_VOCAB = ["Erect", "Recumbent", "UNKNOWN"]

# 只出现过 1 次的 "PA LLD"，统一归到 LLD，避免为它单独开一个 embedding 槽位
_VIEW_ALIAS = {"pa lld": "LLD"}


def _is_missing(value) -> bool:
    """统一判断 None / NaN / 空串。pandas 读空单元格得到的是 NaN(float)。"""
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    return False


def view_index(value) -> int:
    """把 ViewPosition 原始值映射到 VIEW_VOCAB 下标。缺失/异常一律 UNKNOWN。

    保留视位缺失的图像，避免漏掉检查。
    """
    unk = VIEW_VOCAB.index("UNKNOWN")
    if _is_missing(value):
        return unk
    s = str(value).strip()
    if not s:
        return unk
    # 先查别名（小写匹配 "pa lld"），再统一转大写查主词表
    s = _VIEW_ALIAS.get(s.lower(), s.upper())
    return VIEW_VOCAB.index(s) if s in VIEW_VOCAB else unk


def orient_index(value) -> int:
    """把 PatientOrientation 映射到 ORIENT_VOCAB 下标。"""
    unk = ORIENT_VOCAB.index("UNKNOWN")
    if _is_missing(value):
        return unk
    s = str(value).strip()
    if not s:
        return unk
    # 统一成 "Erect" / "Recumbent" 这种首字母大写形式再查表
    s = s[:1].upper() + s[1:].lower()
    return ORIENT_VOCAB.index(s) if s in ORIENT_VOCAB else unk


def parse_predict_class(value) -> np.ndarray:
    """把 '1;8;9' 解析成 (10,) uint8 多热向量。空值/NaN 返回全 0。

    数据事实：`Predict_class` 用分号分隔多标签，每张图 1~7 个。
    这里对非法输入直接抛异常而不是静默忽略——标签错位是最难排查的一类 bug。
    """
    out = np.zeros(NUM_CLASSES, dtype=np.uint8)
    if _is_missing(value):
        return out
    for tok in str(value).split(";"):
        tok = tok.strip()
        if not tok:
            continue        # 容忍 "1;;8" 这类多余分隔符
        if not tok.isdigit():
            raise ValueError(f"Unexpected class token {tok!r} in {value!r}")
        cid = int(tok)
        if not (1 <= cid <= NUM_CLASSES):
            raise ValueError(f"Class id {cid} out of range 1..{NUM_CLASSES}")
        out[cid - 1] = 1    # 1-based id → 0-based 索引
    return out


def parse_many(values: Iterable) -> np.ndarray:
    """解析一列 Predict_class -> (N, 10) uint8。"""
    values = list(values)
    out = np.zeros((len(values), NUM_CLASSES), dtype=np.uint8)
    for i, v in enumerate(values):
        out[i] = parse_predict_class(v)
    return out


def image_relpath(row) -> str:
    """数据集约定的相对路径：{split}/{Subject_id}/{Study_id}/{Dicom_id}.jpg

    实测：文件名（去扩展名）与 CSV 的 Dicom_id 一一对应，全部是 .jpg。
    """
    return os.path.join(str(row["split"]), str(row["Subject_id"]),
                        str(row["Study_id"]), f"{row['Dicom_id']}.jpg")


def load_split_df(split: str, data_root: str, need_labels: bool = True) -> pd.DataFrame:
    """读取某个划分的 CSV，补齐 split / relpath，并可选解析标签。

    Parameters
    ----------
    split : "train" | "valid" | "test"
    data_root : 数据集根目录（含 {split}.csv 与 {split}/ 子目录）
    need_labels : 测试集没有 Predict_class 列，必须传 False
    """
    csv_path = os.path.join(data_root, f"{split}.csv")
    df = pd.read_csv(csv_path, dtype=str)   # 全按字符串读，避免 id 被转成数字丢前导零
    df["split"] = split

    # 前置校验：缺列要在最开始就报错，别等到训练中途
    required = ["Dicom_id", "Subject_id", "Study_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} 缺少必需列: {missing}")

    df["relpath"] = df.apply(image_relpath, axis=1)

    # 尺寸转数值（可能有缺失），用于旁路特征
    for col in ("Rows", "Columns"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 类别型字段转 embedding 下标
    df["view_idx"] = [view_index(v) for v in df.get("ViewPosition", [None] * len(df))]
    df["orient_idx"] = [orient_index(v) for v in df.get("PatientOrientation", [None] * len(df))]

    if need_labels:
        if "Predict_class" not in df.columns:
            raise ValueError(f"{csv_path} 缺少 Predict_class 列（该划分应无标签）")
        df["label"] = list(parse_many(df["Predict_class"].tolist()))
    return df


def labels_matrix(df: pd.DataFrame) -> np.ndarray:
    """从带 label 列的 DataFrame 取 (N,10) uint8。"""
    return np.stack(df["label"].to_numpy()).astype(np.uint8)


def study_table(df: pd.DataFrame) -> pd.DataFrame:
    """聚合到检查级：每个 Study_id 一行。

    实测：同一 Study 内所有图片标签完全一致（18513 个 study，0 冲突）。
    这里仍做一次断言——数据一旦变化会立即暴露，而不是悄悄训出错误模型。
    """
    if "label" not in df.columns:
        raise ValueError("study_table 需要带 label 列的 DataFrame")
    n_unique = df["Study_id"].nunique()
    rows = []
    for sid, g in df.groupby("Study_id", sort=True):
        labs = np.stack(g["label"].to_numpy())
        # 同一检查内所有图必须是同一组标签，否则"标签是检查级的"这个前提不成立
        if len(np.unique(labs, axis=0)) != 1:
            raise AssertionError(f"Study {sid} 内部图片标签不一致，数据假设被打破")
        rows.append({
            "Study_id": sid,
            "Subject_id": g["Subject_id"].iloc[0],
            "n_images": len(g),          # 一个检查有几张图，用于后面选聚合方式
            "label": labs[0],
        })
    out = pd.DataFrame(rows)
    assert len(out) == n_unique
    return out


def image_study_ids(df: pd.DataFrame) -> np.ndarray:
    """图片级 Study_id 数组（用于图像→检查的聚合）。"""
    return df["Study_id"].to_numpy()
