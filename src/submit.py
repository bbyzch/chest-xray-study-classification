"""提交文件生成与校验。

格式（已从赛事官网「七、作品提交要求」核实）：
    Study_id, Predict_class, Probability[, Gate]
    文件命名必须是 submission.csv，编码 UTF-8。

官方原话：
  ● Predict_class：模型预测该次检查对应的各个类别id（**需保证每个 Study_id 下都有
    10 个类别**）→ 所以是**全网格**，每个检查 10 行，共 1414 × 10 = 14140 行。
    这条消除了早期"是否只交 >0.5 的行"的歧义。
  ● Probability：取值范围 [0,1]
  ● Gate：类别阈值默认为 0.5，若需要自定义阈值，填在最后一列，否则该列不填。
    → Gate 只影响**阈值型指标**（Macro-F1），**不影响 Macro-AUC**（AUC 与阈值无关）。
      所以它只在主指标打平、用 F1 二次排序时才起作用。可以用 OOF 上调出的
      逐类别最优阈值填进去，白拿一个 tie-breaker 优势。

注意 Predict_class 是 **1-based**（1..10），内部索引是 0-based，
只有这一层的转换发生在本文件，别处一律用 0..9。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COLUMNS = ["Study_id", "Predict_class", "Probability"]
COLUMNS_WITH_GATE = COLUMNS + ["Gate"]


def build_submission(study_ids, probs: np.ndarray, full_grid: bool = True,
                     threshold: float = 0.5, gates=None) -> pd.DataFrame:
    """将 (S,10) 检查级概率展开为提交长表，类别ID从内部0-based转为1-based。

    full_grid=True 时每个检查输出10行；稀疏模式保留达到 threshold 的行。
    gates 可为标量或长度10的阈值，None 时不生成 Gate 列。"""
    study_ids = np.asarray(study_ids)
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[1] != 10:
        raise ValueError(f"probs 需要形状 (S,10), 得到 {probs.shape}")
    if len(study_ids) != probs.shape[0]:
        raise ValueError("study_ids 与 probs 行数不一致")

    if gates is not None:
        gates = np.broadcast_to(np.asarray(gates, dtype=np.float64), (10,))
        if not np.all((gates >= 0) & (gates <= 1)):
            raise ValueError("Gate 阈值必须在 [0,1]")

    submission_rows = []
    for sid, p in zip(study_ids, probs):
        for c in range(10):
            if full_grid or p[c] >= threshold:
                # c 是 0-based 索引，提交要 1-based 类别 id，所以 +1
                submission_row = (sid, c + 1, float(p[c]))
                if gates is not None:
                    submission_row = submission_row + (float(gates[c]),)
                submission_rows.append(submission_row)

    columns = COLUMNS_WITH_GATE if gates is not None else COLUMNS
    return pd.DataFrame(submission_rows, columns=columns)


def validate_submission(sub: pd.DataFrame, expected_study_ids,
                        full_grid: bool = True) -> None:
    """提交前的硬断言。失败即抛异常，绝不让坏文件出门。

    每天只有 2 次提交额度，一次格式错误就是半天白等，所以这里宁严勿松。
    """
    # 1) 必需的三个列必须在（Gate 可选，不强制）
    missing = [c for c in COLUMNS if c not in sub.columns]
    if missing:
        raise ValueError(f"缺少列: {missing}")

    # 2) Study_id 集合必须与测试集**完全一致**：少了会漏检查，多了会被判无效
    submitted_studies = set(sub["Study_id"].astype(str))
    expected_studies = set(np.asarray(expected_study_ids).astype(str))
    if submitted_studies != expected_studies:
        missing_studies = sorted(expected_studies - submitted_studies)[:5]
        extra_studies = sorted(submitted_studies - expected_studies)[:5]
        raise ValueError(
            f"Study_id 集合不匹配：缺 {len(expected_studies - submitted_studies)} 个 {missing_studies}，"
            f"多 {len(submitted_studies - expected_studies)} 个 {extra_studies}")

    # 3) 类别 id 必须落在 1..10
    if not sub["Predict_class"].isin(range(1, 11)).all():
        bad = sorted(set(sub["Predict_class"]) - set(range(1, 11)))
        raise ValueError(f"Predict_class 越界: {bad}")

    # 4) 概率必须是有限值且在 [0,1]
    prob = sub["Probability"].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(prob)):
        raise ValueError("Probability 含 NaN/Inf")
    if prob.min() < 0 or prob.max() > 1:
        raise ValueError(f"Probability 越界: [{prob.min()}, {prob.max()}]")

    # 5) Gate 若存在，必须全为 [0,1] 内的有限值（官方：不填就用默认 0.5）
    if "Gate" in sub.columns:
        gate = sub["Gate"].to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(gate)):
            raise ValueError("Gate 含 NaN/Inf（不确定就整列删除，而不是填空）")
        if gate.min() < 0 or gate.max() > 1:
            raise ValueError(f"Gate 越界: [{gate.min()}, {gate.max()}]")

    # 6) 官方要求每个 Study_id 下必须恰好 10 个类别
    if full_grid:
        class_counts = sub.groupby("Study_id")["Predict_class"].nunique()
        bad = class_counts[class_counts != 10]
        if len(bad):
            raise ValueError(f"{len(bad)} 个 Study_id 的类别数不等于 10，"
                             f"示例 {bad.head().to_dict()}")
        if sub.duplicated(["Study_id", "Predict_class"]).any():
            raise ValueError("存在重复的 (Study_id, Predict_class)")


def write_submission(sub: pd.DataFrame, path: str, expected_study_ids=None,
                     full_grid: bool = True) -> str:
    """校验并落盘。

    - 文件名按官方要求应为 submission.csv
    - 编码固定 UTF-8，且**不写 BOM**（utf-8-sig 会让首列名变成 "\\ufeffStudy_id"，
      某些评测脚本会因此找不到列）
    - 固定 6 位小数：AUC 只看排序，6 位足够，且能显著减小文件体积
    """
    if expected_study_ids is not None:
        validate_submission(sub, expected_study_ids, full_grid=full_grid)
    sub.to_csv(path, index=False, float_format="%.6f", encoding="utf-8")
    return path


def tune_gates(y_true: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """在 OOF 上按类别搜索使 Macro-F1 最大的阈值，用于填 Gate 列。

    只在主指标（Macro-AUC）打平、靠 Macro-F1 二次排序时才有意义，
    但成本极低，属于"白拿"的收益。
    """
    y = np.asarray(y_true)
    s = np.asarray(probs, dtype=np.float64)
    gates = np.full(y.shape[1], 0.5, dtype=np.float64)

    for c in range(y.shape[1]):
        class_labels = y[:, c]
        if class_labels.sum() == 0 or class_labels.sum() == len(class_labels):
            continue                    # 退化类别，保持 0.5
        best_class_f1, best_class_gate = -1.0, 0.5
        for t in np.linspace(0.02, 0.98, 97):
            pred = s[:, c] >= t
            tp = float(np.sum(pred & (class_labels == 1)))
            fp = float(np.sum(pred & (class_labels == 0)))
            fn = float(np.sum(~pred & (class_labels == 1)))
            if tp == 0:
                continue
            f1 = 2 * tp / (2 * tp + fp + fn)
            if f1 > best_class_f1:
                best_class_f1, best_class_gate = f1, float(t)
        gates[c] = best_class_gate
    return gates
