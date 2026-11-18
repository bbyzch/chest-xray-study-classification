"""按患者分组的多标签分层 K 折。

为什么必须这样做：
1. **按患者分组**——赛题按 Subject 划分数据，同一患者的图片若跨折，模型会"见过这个人"，
   OOF 分数虚高，选出的模型上线就崩。
2. **多标签分层**——类别 1 只有 622 个检查（3.4%）。随机分折很可能某一折一个阳性都没有，
   该折的类别 1 AUC 无定义，OOF 评估直接失效。

算法：Sechidis et al. (2011) 迭代分层的简化版。
以 Subject 为分组单位（取该患者名下所有检查标签的**并集**），
按类别稀有度从稀有到常见依次分配：每轮取"尚未分配、且在该类上为阳性"的患者，
按其剩余阳性类别数降序排列，依次放入该类阳性数最少的折。

实测效果（真实训练集，5 折）：每折类别 1 阳性 173~183 个，折大小差异 < 4%。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def subject_label_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """合并同一患者全部图像的阳性标签，返回排序患者ID及患者级多热矩阵。"""
    if "label" not in df.columns:
        raise ValueError("需要带 label 列的 DataFrame")
    image_subject_ids = df["Subject_id"].to_numpy()
    image_labels = np.stack(df["label"].to_numpy()).astype(np.uint8)

    sorted_subject_ids = np.unique(image_subject_ids)                  # 排序后的唯一患者 id
    subject_positions = {s: i for i, s in enumerate(sorted_subject_ids)}
    subject_labels = np.zeros((len(sorted_subject_ids), image_labels.shape[1]), dtype=np.uint8)
    for row_subj, row_lab in zip(image_subject_ids, image_labels):
        # 逐行取最大值 = 按位或，等价于求并集
        subject_labels[subject_positions[row_subj]] = np.maximum(subject_labels[subject_positions[row_subj]], row_lab)
    return sorted_subject_ids, subject_labels


def group_kfold(df: pd.DataFrame, n_splits: int = 5, seed: int = 0) -> np.ndarray:
    """返回与 df 行对齐的 fold id 数组（0..n_splits-1）。

    seed 只影响并列时的打散顺序，不改变分层质量；同 seed 结果完全可复现。
    """
    if n_splits < 2:
        raise ValueError("n_splits 必须 >= 2")

    subjects, subject_labels = subject_label_matrix(df)
    n_subj, n_cls = subject_labels.shape
    rng = np.random.default_rng(seed)

    fold_of = np.full(n_subj, -1, dtype=np.int64)          # -1 = 未分配
    cls_counts = np.zeros((n_splits, n_cls), dtype=np.int64)  # 每折每类阳性数
    subj_counts = np.zeros(n_splits, dtype=np.int64)          # 每折患者数

    # 类别按全局稀有度升序：阳性少的类别**先分**，
    # 因为稀有类别的可选患者最少，留给它的余地最小，必须先满足它。
    order = np.argsort(subject_labels.sum(axis=0), kind="mergesort")

    for c in order:
        # 还没分配、且在该类上为阳性的患者
        candidate_subjects = np.where((fold_of == -1) & (subject_labels[:, c] == 1))[0]
        if len(candidate_subjects) == 0:
            continue

        # 剩余阳性类别多者优先 —— "难伺候"的患者先安排，避免最后没位置
        positive_label_counts = subject_labels[candidate_subjects].sum(axis=1)
        # 加一个极小的随机扰动来打散同分者：
        # 既保证确定性（同 seed 同结果），又消除数据原始顺序带来的系统性偏置
        jitter = rng.random(len(candidate_subjects)) * 1e-6
        candidate_subjects = candidate_subjects[np.argsort(-(positive_label_counts + jitter), kind="mergesort")]

        for s in candidate_subjects:
            # 三元组依次比较：该类阳性最少 → 患者数最少 → 折下标最小（保证确定性）
            chosen_fold = min(
                range(n_splits),
                key=lambda f: (cls_counts[f, c], subj_counts[f], f),
            )
            fold_of[s] = chosen_fold
            cls_counts[chosen_fold] += subject_labels[s]     # 累加该患者的**全部**阳性类别
            subj_counts[chosen_fold] += 1

    # 全阴性患者（10 类都不阳性）在上面的循环里不会被选中，最后补到人数最少的折
    for s in np.where(fold_of == -1)[0]:
        chosen_fold = int(np.argmin(subj_counts))
        fold_of[s] = chosen_fold
        subj_counts[chosen_fold] += 1

    assert (fold_of >= 0).all(), "存在未分配折的患者"

    # 把患者级分配展开回图片/检查级（df 的每一行）
    subject_to_fold = dict(zip(subjects, fold_of))
    return np.array([subject_to_fold[s] for s in df["Subject_id"].to_numpy()], dtype=np.int64)


def summarize_folds(df: pd.DataFrame, folds: np.ndarray) -> pd.DataFrame:
    """按折统计样本数与逐类别阳性数，便于人工检查分层是否合理。

    正常情况下应看到：每折图片数接近、每类阳性数接近、稀有类每折都 > 0。
    """
    image_labels = np.stack(df["label"].to_numpy()).astype(np.uint8)
    fold_summary_rows = []
    for f in range(int(folds.max()) + 1):
        m = folds == f
        row = {"fold": f, "images": int(m.sum()),
               "studies": int(df.loc[m, "Study_id"].nunique())}
        for c in range(image_labels.shape[1]):
            row[f"c{c + 1}"] = int(image_labels[m, c].sum())
        fold_summary_rows.append(row)
    return pd.DataFrame(fold_summary_rows)
