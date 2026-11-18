"""评价指标：纯 numpy 实现的 ROC-AUC 与宏平均。

不依赖 sklearn —— 一是避免给国赛容器增加依赖，二是关键算法自实现后可以被精确测试
（实测：与 O(n²) 两两比较的朴素定义在 200 组随机数据上逐项一致）。

AUC 的两种等价定义：
  1. 概率论：随机取一个正样本和一个负样本，前者分数更高的概率
  2. 秩和（Mann-Whitney U）：本文件采用，O(n log n)
分数并列时按 (1 胜 + 0.5 平) 计，对应"平均秩"处理。
"""
from __future__ import annotations

import numpy as np


def rankdata(a: np.ndarray) -> np.ndarray:
    """计算1-based平均秩，并列预测值共享其所在秩区间的均值。"""
    a = np.asarray(a, dtype=np.float64)
    unique_scores, score_to_rank_group, counts = np.unique(a, return_inverse=True, return_counts=True)
    rank_upper_bounds = np.cumsum(counts).astype(np.float64)   # 每组右端点（1-based 秩）
    avg_rank = rank_upper_bounds - (counts - 1) / 2.0          # 组内秩的均值
    return avg_rank[score_to_rank_group].astype(np.float64)


def roc_auc(y_true, y_score) -> float:
    """二分类 AUC。若某一类样本缺失（全正或全负），返回 nan。

    秩和公式：
        AUC = (正样本秩和 - n_pos(n_pos+1)/2) / (n_pos · n_neg)
    其中减去的项是所有秩和的最小可能值（正样本全排最前时的秩和）。
    """
    y = np.asarray(y_true).ravel()
    s = np.asarray(y_score, dtype=np.float64).ravel()
    if y.shape != s.shape:
        raise ValueError(f"shape mismatch: {y.shape} vs {s.shape}")
    if not np.all(np.isin(y, (0, 1))):
        raise ValueError("y_true 必须是 0/1")

    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    # 单类缺失时 AUC 无定义。这里返回 nan 而不是 0/1，
    # 让 macro_auc 能识别并跳过该类别。
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    if not np.all(np.isfinite(s)):
        raise ValueError("y_score 含 NaN/Inf")

    ranks = rankdata(s)
    auc = (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def macro_auc(y_true, y_score) -> tuple[float, dict[int, float]]:
    """计算检查级多标签宏平均AUC，输入为同形状的 (N,C) 标签和分数。

    返回宏平均及0-based逐类别AUC；仅对同时含正负样本的类别取平均。
    这里显式跳过退化类别，不等同于 sklearn 所有输入情况下的默认行为。"""
    y = np.asarray(y_true)
    s = np.asarray(y_score, dtype=np.float64)
    if y.ndim != 2 or s.ndim != 2 or y.shape != s.shape:
        raise ValueError(f"期望形状一致的两个二维数组, 得到 {y.shape} 与 {s.shape}")

    per_class: dict[int, float] = {}
    valid_class_aucs: list[float] = []
    for c in range(y.shape[1]):
        auc = roc_auc(y[:, c], s[:, c])
        per_class[c] = auc
        if not np.isnan(auc):
            valid_class_aucs.append(auc)
    macro = float(np.mean(valid_class_aucs)) if valid_class_aucs else float("nan")
    return macro, per_class


def bootstrap_macro_auc(y_true, y_score, n_boot: int = 200,
                        seed: int = 0) -> tuple[float, float]:
    """对检查做 bootstrap 重采样，返回宏平均 AUC 的 (2.5%, 97.5%) 分位区间。

    用途：判断两个配置的差距是否超过噪声。设计文档 §5 的纪律是
    「差异 < 0.005 视为噪声，不采纳」——这个函数就是那条纪律的量化依据。
    """
    y = np.asarray(y_true)
    s = np.asarray(y_score, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(y)
    bootstrap_aucs = []
    for _ in range(n_boot):
        # 有放回抽样，模拟"换一批测试检查"
        resampled_studies = rng.integers(0, n, size=n)
        m, _ = macro_auc(y[resampled_studies], s[resampled_studies])
        if not np.isnan(m):
            bootstrap_aucs.append(m)
    if not bootstrap_aucs:
        return float("nan"), float("nan")
    lo, hi = np.percentile(bootstrap_aucs, [2.5, 97.5])
    return float(lo), float(hi)
