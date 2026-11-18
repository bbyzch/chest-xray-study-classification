"""图像级概率 → 检查级概率，以及类别 10 融合。

赛题评估在**检查级**（Study_id），而模型是图像级的，必须聚合。
同一检查有多张图（train 中位 2 张，最多 7 张）。

类别 10 的融合是本方案的核心增益点（设计文档 §2.3 / §4.4）：
实测类别 10「无异常」与类别 1–9 **零共现**（12649 个含 10 的样本，0 冲突），
所以除了独立的第 10 个头，还可以用 ∏(1-p_i) 作为第二个信号，两者加权融合。
宏平均 AUC 里类别 10 占 1/10 权重，而多数参赛队会照搬"10 个独立 sigmoid"。

本模块是纯 numpy，不依赖 torch，可脱离 GPU 完整测试。
"""
from __future__ import annotations

import numpy as np

# 概率裁剪下限：log / 除法前必须避开 0 和 1，否则得到 inf 并污染整个分数
EPS = 1e-6
DEFAULT_STUDY_AGGREGATION = "mean"


def _logit(p: np.ndarray, eps: float = EPS) -> np.ndarray:
    """logit 变换：把 [0,1] 的分数映射到 (-inf, +inf)，让 max/mean 这类
    线性运算在"证据强度"空间而不是概率空间进行。"""
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """logit 的逆变换。"""
    return 1.0 / (1.0 + np.exp(-x))


def aggregate_study(img_probs: np.ndarray, study_ids: np.ndarray,
                    mode: str = DEFAULT_STUDY_AGGREGATION, temperature: float = 1.0
                    ) -> tuple[np.ndarray, np.ndarray]:
    """按 Study_id 聚合多视位概率，支持 mean、max 与计数修正的 logsumexp。

    输入为 (N,C) 概率及 N 个检查ID；返回按ID排序的 (S,C) 概率和检查ID。
    temperature 仅作用于 logsumexp，默认 mean 与提交推理保持一致。"""
    img_probs = np.asarray(img_probs, dtype=np.float64)
    study_ids = np.asarray(study_ids)
    if img_probs.ndim != 2:
        raise ValueError(f"img_probs 需要二维, 得到 {img_probs.shape}")
    if len(img_probs) != len(study_ids):
        raise ValueError("img_probs 与 study_ids 长度不一致")

    sorted_study_ids, image_to_study = np.unique(study_ids, return_inverse=True)  # inv: 每行属于第几个检查
    n_study, n_cls = len(sorted_study_ids), img_probs.shape[1]
    study_scores = np.zeros((n_study, n_cls), dtype=np.float64)

    if mode == "max":
        # np.maximum.at 支持重复下标（普通索引赋值是"最后写入者获胜"而非取最大）
        np.maximum.at(study_scores, image_to_study, img_probs)

    elif mode == "mean":
        np.add.at(study_scores, image_to_study, img_probs)                      # 先求和
        study_scores /= np.bincount(image_to_study, minlength=n_study)[:, None]  # 再除以该检查的图片数

    elif mode == "logsumexp":
        if temperature <= 0:
            raise ValueError("temperature 必须 > 0")
        z = _logit(img_probs) / temperature   # 除以 T：T 大 → 各项接近，趋于均值
        for s in range(n_study):
            m = image_to_study == s
            study_logits = z[m]
            n = study_logits.shape[0]
            # 数值稳定的 logsumexp：先减去该列最大值再取指数，避免溢出
            peak_logits = study_logits.max(axis=0)
            pooled_logits = peak_logits + np.log(np.exp(study_logits - peak_logits).sum(axis=0))
            # 减去 log(n) 做**计数修正**：否则图片多的检查天然分数更高，
            # 而检查的图片数从 1 到 7 不等，这会变成一个纯噪声特征。
            study_scores[s] = _sigmoid((pooled_logits - np.log(n)) * temperature)
    else:
        raise ValueError(f"未知聚合方式 {mode!r}")

    return study_scores, sorted_study_ids


def blend_no_finding(study_probs: np.ndarray, w_head: float = 1.0,
                     w_product: float = 1.0) -> np.ndarray:
    """类别 10 融合：w_head · p10_头 + w_product · ∏(1-p_i)。

    返回新数组，第 9 列（索引 9，即类别 10）被替换，其余列不变。

    直觉：如果 9 个病灶的概率都很低，∏(1-p_i) 接近 1（确实无异常）；
    只要有一个病灶概率高，乘积就会被显著压低。这正好编码了"互斥"这个先验。
    """
    probs = np.asarray(study_probs, dtype=np.float64)
    if probs.shape[1] != 10:
        raise ValueError(f"需要 10 类, 得到 {probs.shape}")

    study_scores = probs.copy()
    if w_product == 0:
        return study_scores          # 只用独立头，走快速路径
    finding_probs = np.clip(probs[:, :9], EPS, 1.0 - EPS)
    no_finding_product = np.prod(1.0 - finding_probs, axis=1)     # ∏(1-p_i)，i 取 1..9
    study_scores[:, 9] = w_head * probs[:, 9] + w_product * no_finding_product
    return study_scores


def parse_patient_smooth(spec) -> np.ndarray | None:
    """解析患者平滑系数：None/"" → None；单个数 → 10 类同值；10 个逗号分隔数 → 逐类。"""
    if spec is None or (isinstance(spec, str) and not spec.strip()):
        return None
    if isinstance(spec, str):
        vals = [float(x) for x in spec.split(",") if x.strip()]
    else:
        vals = list(np.atleast_1d(spec).astype(float))
    if len(vals) == 1:
        vals = vals * 10
    if len(vals) != 10:
        raise ValueError(f"患者平滑系数需要 1 个或 10 个值，得到 {len(vals)} 个")
    alphas = np.asarray(vals, dtype=np.float64)
    if np.any(alphas < 0) or np.any(alphas > 1):
        raise ValueError("患者平滑系数必须在 [0, 1]")
    return alphas


def patient_smooth(study_probs: np.ndarray, subject_ids: np.ndarray,
                   alphas) -> np.ndarray:
    """患者平滑：p' = (1-α)·p_本检查 + α·mean(同一患者**其他**检查的 p)。

    合规依据：《补充说明》第七条（一）—— 对同一患者多个测试检查的前向输出做
    确定性聚合，不拟合参数、不计算测试集统计量、不用测试标签。α 在训练集
    OOF 上按留一折选定后写死，推理时不调整。患者分组来自官方 Subject_id。

    只有一个检查的患者保持原值。alphas 为长度 10 的逐类系数（或标量）。
    """
    probs = np.asarray(study_probs, dtype=np.float64)
    subj = np.asarray(subject_ids)
    if probs.ndim != 2 or len(subj) != len(probs):
        raise ValueError(f"形状不匹配: probs {probs.shape} vs subject_ids {subj.shape}")
    a = np.broadcast_to(np.asarray(alphas, dtype=np.float64), (probs.shape[1],))

    _, inv, counts = np.unique(subj, return_inverse=True, return_counts=True)
    sums = np.zeros((len(counts), probs.shape[1]))
    np.add.at(sums, inv, probs)
    n = counts[inv][:, None].astype(np.float64)
    others = np.where(n > 1, (sums[inv] - probs) / np.maximum(n - 1, 1), probs)
    return (1.0 - a) * probs + a * others


def fit_no_finding_weight(y_true: np.ndarray, study_probs: np.ndarray,
                          grid: int = 41) -> dict:
    """在 OOF 上搜索类别10独立头与病灶互斥乘积的融合权重。

    返回 w_head、w_product 和 auc；单类缺失时保留独立头初始值。"""
    from src.metrics import roc_auc

    y10 = np.asarray(y_true)[:, 9]
    head = np.asarray(study_probs)[:, 9]
    no_finding_product = np.prod(1.0 - np.clip(study_probs[:, :9], EPS, 1.0 - EPS), axis=1)

    best_blend = {"w_head": 1.0, "w_product": 0.0, "auc": roc_auc(y10, head)}
    if np.isnan(best_blend["auc"]):
        return best_blend             # 退化情形：直接返回，不做无意义的搜索
    for w in np.linspace(0.0, 1.0, grid):
        blended_score = w * head + (1.0 - w) * no_finding_product
        auc = roc_auc(y10, blended_score)
        if not np.isnan(auc) and auc > best_blend["auc"]:
            best_blend = {"w_head": float(w), "w_product": float(1.0 - w), "auc": float(auc)}
    return best_blend
