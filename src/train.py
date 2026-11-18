"""单折训练 + OOF 产出。

产物（runs/<exp>/fold{k}/）：
    ckpt.pt      最佳 epoch 的 EMA 权重（只存 state_dict，加载端 weights_only=True）
    oof.npz      留出折的图像级概率 + row_idx + study_id + 标签
    metrics.json 逐 epoch 曲线与最佳宏 AUC

评估口径一律是**检查级**宏平均 AUC，默认聚合方式与提交推理一致（mean）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.aggregate import DEFAULT_STUDY_AGGREGATION  # noqa: E402
from src.config import TrainConfig  # noqa: E402
from src.dataset import CXRMemmapDataset  # noqa: E402
from src.labels import VIEW_VOCAB  # noqa: E402
from src.losses import (EMA, amp_dtype, asl_multilabel, bce_multilabel,
                        cosine_lr, distill_multilabel, l1_penalty)  # noqa: E402
from src.metrics import macro_auc  # noqa: E402
from src.models import build_model  # noqa: E402
from src.splits import group_kfold  # noqa: E402

LATERAL_IDX = VIEW_VOCAB.index("LATERAL")


class StudyBatchSampler(Sampler[list[int]]):
    """按 Study 组成 batch，保证一个检查的所有图像一起进入模型。

    ``max_images`` 仍按图像数限制显存；每个 Study 只有 1~7 张图，
    因此一个 batch 通常包含多个完整检查。这样训练损失可以直接在
    检查级聚合后计算，与验证和提交的 mean 聚合口径一致。
    """

    def __init__(self, study_idx: np.ndarray, max_images: int, seed: int,
                 shuffle: bool = True):
        self.max_images = int(max_images)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        groups = []
        for sid in np.unique(study_idx):
            groups.append(np.flatnonzero(study_idx == sid).astype(np.int64).tolist())
        self.groups = groups
        self.epoch = 0

    def __len__(self) -> int:
        n = 0
        images = 0
        for group in self.groups:
            if images and images + len(group) > self.max_images:
                n += 1
                images = 0
            images += len(group)
        return n + bool(images)

    def __iter__(self):
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            self.epoch += 1
            order = torch.randperm(len(self.groups), generator=generator).tolist()
        else:
            order = list(range(len(self.groups)))
        batch, images = [], 0
        for group_id in order:
            group = self.groups[group_id]
            if batch and images + len(group) > self.max_images:
                yield batch
                batch, images = [], 0
            batch.extend(group)
            images += len(group)
        if batch:
            yield batch


def aggregate_batch_studies(logits: torch.Tensor, study_idx: torch.Tensor):
    """把按检查连续排列的图像 logits 聚合成检查级 logits/targets 索引。

    先平均 sigmoid 概率，再映射回 logits，和提交阶段的 mean 聚合保持一致。
    ``first`` 是每个 Study 的首图位置，用于取重复标签和样本权重。
    """
    if study_idx.ndim != 1 or logits.shape[0] != study_idx.shape[0]:
        raise ValueError("logits 与 study_idx 的 batch 维度不一致")
    starts = torch.ones_like(study_idx, dtype=torch.bool)
    if len(study_idx) > 1:
        starts[1:] = study_idx[1:] != study_idx[:-1]
    first = torch.nonzero(starts, as_tuple=False).flatten()
    group = torch.cumsum(starts.to(torch.long), dim=0) - 1
    probs = torch.sigmoid(logits.float())
    sums = torch.zeros((len(first), logits.shape[1]), device=logits.device)
    sums.index_add_(0, group, probs)
    counts = torch.bincount(group, minlength=len(first)).to(logits.device)
    mean_probs = sums / counts[:, None].clamp_min(1)
    agg_logits = torch.logit(mean_probs.clamp(1e-5, 1.0 - 1e-5))
    return agg_logits, first, group, counts


def fold_frame(meta, labels):
    """只用**缓存里的信息**构造分折所需的表。

    刻意不读原始 train.csv：云端机器上只有 cache/，没有 55GB 原始数据集，
    一旦依赖 CSV 就直接跑不起来。index_train.csv 已带 Subject_id/Study_id，
    train_labels.npy 带标签，足够做分层分组 K 折。
    """
    import pandas as pd

    return pd.DataFrame({
        "Subject_id": meta["Subject_id"].to_numpy(),
        "Study_id": meta["Study_id"].to_numpy(),
        "label": list(labels),
    })


def load_cache_index(cfg: TrainConfig):
    import pandas as pd

    meta = pd.read_csv(os.path.join(cfg.cache_dir, "index_train.csv"))
    labels = np.load(os.path.join(cfg.cache_dir, "train_labels.npy"))
    with open(os.path.join(cfg.cache_dir, "norm.json"), encoding="utf-8") as f:
        norm = json.load(f)
    if cfg.limit_images:
        meta = meta.head(cfg.limit_images).reset_index(drop=True)
        labels = labels[:cfg.limit_images]
    return meta, labels, norm


def load_distill_targets(path: str | None, n_rows: int):
    """加载并验证按全局 train row_idx 排列的 OOF 教师软目标。"""
    if not path:
        return None
    d = np.load(path, allow_pickle=False)
    rows = d["row_idx"].astype(np.int64)
    probs = d["teacher_probs"].astype(np.float32)
    if (probs.ndim != 2 or probs.shape[1] != 10 or len(probs) < n_rows
            or not np.array_equal(rows[:n_rows], np.arange(n_rows, dtype=np.int64))):
        raise ValueError(
            f"蒸馏目标必须覆盖 train row_idx 0..{n_rows - 1}，得到 rows={rows.shape}, "
            f"probs={probs.shape}")
    return np.clip(probs[:n_rows], 1e-5, 1.0 - 1e-5)


def make_dataset(cfg: TrainConfig, meta, labels, norm, train: bool) -> CXRMemmapDataset:
    return CXRMemmapDataset(
        arr_path=os.path.join(cfg.cache_dir, f"train_{cfg.cache_size}.npy"),
        rows=meta,
        image_size=cfg.image_size,
        train=train,
        mean=norm["mean"],
        std=norm["std"],
        labels=labels,
        use_clahe=cfg.use_clahe,
        size_mean=norm["size_mean"],
        size_std=norm["size_std"],
        cache_size=cfg.cache_size,
        aug=cfg.aug,
    )


def load_conflict_rows(path: str | None, meta, labels: np.ndarray) -> np.ndarray:
    """把 find_conflicts.py 导出的冲突检查，映射成**图像行的布尔掩码**。

    find_conflicts 是检查级的（它导出的 csv 有 study_id），而训练是图像级的，
    所以要把「哪些检查有标签冲突」翻译成「哪些图要降权」。

    为什么要按 study_id 而不是直接用 row_idx：csv 里的 row_idx 是**检查级**的
    排序位置（aggregate_study 输出的顺序），和训练用的图像级 row_idx 不是一回事，
    直接拿来用会错位。按 study_id 映射既安全又不需要额外的对照表。

    同一检查的所有图片共享标签，所以这些图**全部**降权——标签错了，
    该检查的每一张图都带着错标签。
    """
    if not path:
        return np.zeros(len(meta), dtype=bool)
    import pandas as pd

    df = pd.read_csv(path)
    if "study_id" not in df.columns:
        raise ValueError(
            f"{path} 缺少 study_id 列。请用 find_conflicts.py --export 重新导出"
            f"（旧版导出的是图像级 row_idx，无法可靠映射）")
    conflict_studies = set(df["study_id"].astype(str).unique())
    mask = meta["Study_id"].astype(str).isin(conflict_studies).to_numpy()
    return mask.astype(bool)


def sample_weights(view_idx: torch.Tensor, lateral_weight: float) -> torch.Tensor:
    """按视位给逐样本权重：侧位图降权，其余为 1.0。

    为什么必须从 **batch 自身的 view_idx** 算，而不能用预先算好的数组按位置切片：
    DataLoader 开了 shuffle，batch 内的样本顺序与数据集顺序不一致，
    按位置切片会把侧位权重贴到错误样本上（而且不会报错，只是悄悄训歪）。
    """
    return torch.where(view_idx == LATERAL_IDX, float(lateral_weight), 1.0)


def study_count_weights(study_ids: np.ndarray) -> np.ndarray:
    """按检查归一化：每张图的权重 = 1 / 该检查的图片数。

    为什么需要：比赛指标与标签都是**检查级**的，但训练是**图像级**——
    同一检查的标签被复制到每张图上，逐图算损失就等于该检查贡献了 N 次梯度
    （N = 该检查的图片数）。于是「4 张图的检查」比「1 张图的检查」权重大 4 倍，
    而评估时两者等权。这会造成训练目标与评测口径不一致。

    乘以 1/N 后，**每个检查对总损失的贡献恒等**（设图片权重之和 = 1），
    与它有几张图无关。这是 MIL 的简化版：不改模型结构，只对齐损失权重。

    注意：这与侧位降权是**两个独立维度**，乘法叠加：
        最终权重 = (1 / 检查图片数) × 侧位系数

    返回按传入顺序的 float32 数组。
    """
    ids = np.asarray(study_ids, dtype=str)
    uniq, inv, cnt = np.unique(ids, return_inverse=True, return_counts=True)
    return (1.0 / cnt[inv]).astype(np.float32)


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """返回 (图像级概率 (N,10), row_idx (N,))。"""
    model.eval()
    probs, rows = [], []
    for batch in loader:
        logits = model(batch["image"].to(device, non_blocking=True),
                       batch["view_idx"].to(device, non_blocking=True),
                       batch["orient_idx"].to(device, non_blocking=True),
                       batch["size_feat"].to(device, non_blocking=True))
        probs.append(torch.sigmoid(logits.float()).cpu().numpy())
        rows.append(batch["row_idx"].numpy())
    return np.concatenate(probs), np.concatenate(rows)


@torch.no_grad()
def evaluate_chex_fusion(model, loader, device):
    """返回 Study 融合概率，同时保留图像概率用于兼容 OOF 产物。"""
    model.eval()
    study_probs, study_rows, image_probs, image_rows = [], [], [], []
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        view = batch["view_idx"].to(device, non_blocking=True)
        orient = batch["orient_idx"].to(device, non_blocking=True)
        size = batch["size_feat"].to(device, non_blocking=True)
        study_idx = batch["study_idx"].to(device, non_blocking=True)
        study_logits, first = model.forward_study(
            image, view, orient, size, study_idx)
        image_logits = model(image, view, orient, size)
        study_probs.append(torch.sigmoid(study_logits.float()).cpu().numpy())
        study_rows.append(batch["row_idx"][first.cpu()].numpy())
        image_probs.append(torch.sigmoid(image_logits.float()).cpu().numpy())
        image_rows.append(batch["row_idx"].numpy())
    return (np.concatenate(study_probs), np.concatenate(study_rows),
            np.concatenate(image_probs), np.concatenate(image_rows))


def study_level_auc(img_probs: np.ndarray, row_idx: np.ndarray, meta, labels,
                    mode: str = DEFAULT_STUDY_AGGREGATION):
    """图像概率 → 按提交默认口径聚合到检查级 → 宏平均 AUC。

    注意 row_idx 是**全量缓存**的行号（dataset 里存在 row_idx 列，跨折唯一），
    所以这里直接索引全量 meta / labels，不要用折内子集去索引。
    """
    from src.aggregate import aggregate_study

    sid = meta["Study_id"].to_numpy()[row_idx]
    study_probs, uniq = aggregate_study(img_probs, sid, mode=mode)
    first = {}
    for r, s in zip(row_idx, sid):
        first.setdefault(s, labels[int(r)])   # 同一 study 内标签一致
    y = np.stack([first[s] for s in uniq])
    return macro_auc(y, study_probs)


def train_fold(cfg: TrainConfig, fold: int, folds: np.ndarray,
               meta, labels, norm, device: torch.device, verbose: bool = True) -> dict:
    # 初始化种子与数据划分种子分开：做多 seed 集成时，分折必须一致（用 seed），
    # 只改 init_seed。否则 OOF 覆盖的检查不同，集成工具会判为不一致而跳过。
    init_seed = cfg.seed if cfg.init_seed is None else cfg.init_seed
    torch.manual_seed(init_seed + fold)
    np.random.seed(init_seed + fold)

    # 按折切分：tr_sel / va_sel 是对全量 meta 的布尔掩码。
    # meta_va 只在构造 dataset 时用；评估时仍用全量 meta 索引（row_idx 是全局行号）。
    tr_sel, va_sel = folds != fold, folds == fold
    meta_tr = meta[tr_sel].reset_index(drop=True)
    meta_va = meta[va_sel].reset_index(drop=True)
    lab_tr, lab_va = labels[tr_sel], labels[va_sel]
    teacher_all = load_distill_targets(cfg.distill_targets, len(labels))

    ds_tr = make_dataset(cfg, meta_tr, lab_tr, norm, train=True)
    ds_va = make_dataset(cfg, meta_va, lab_va, norm, train=False)

    nw = cfg.num_workers
    # drop_last：训练时丢弃最后一个不完整 batch，避免 BN 统计量被小 batch 带偏。
    # 但数据集比 batch 还小时必须关掉，否则一个 batch 都没有。
    if cfg.study_mil or cfg.chex_fusion:
        study_sampler = StudyBatchSampler(ds_tr.study_idx, cfg.batch_size,
                                          seed=init_seed + fold)
        dl_tr = DataLoader(ds_tr, batch_sampler=study_sampler, num_workers=nw,
                           pin_memory=(device.type == "cuda"),
                           persistent_workers=nw > 0)
        if verbose:
            print(f"  Study grouping: {len(study_sampler.groups)} 个检查，"
                  f"每批最多 {cfg.batch_size} 张图", flush=True)
    else:
        dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True, num_workers=nw,
                           pin_memory=(device.type == "cuda"), drop_last=len(ds_tr) > cfg.batch_size,
                           persistent_workers=nw > 0)
    dl_va = DataLoader(ds_va, batch_size=cfg.batch_size, shuffle=False, num_workers=nw,
                       pin_memory=(device.type == "cuda"), persistent_workers=nw > 0)
    dl_va_fusion = None
    if cfg.chex_fusion:
        va_sampler = StudyBatchSampler(ds_va.study_idx, cfg.batch_size,
                                       seed=init_seed + fold, shuffle=False)
        dl_va_fusion = DataLoader(ds_va, batch_sampler=va_sampler, num_workers=nw,
                                  pin_memory=(device.type == "cuda"),
                                  persistent_workers=nw > 0)

    # 两个模型实例：model 负责训练，eval_model 只用来灌 EMA 权重做评估。
    # eval_model 用 pretrained=False 构造（省一次权重下载），结构完全一致。
    # pool / label_graph 必须两边都传，否则 load_state_dict 会因为键不匹配而报错。
    model_kw = dict(num_view=len(VIEW_VOCAB), dropout=cfg.dropout,
                    view_dim=cfg.view_dim, orient_dim=cfg.orient_dim,
                    hidden=cfg.hidden, pool=cfg.pool, csra_lambda=cfg.csra_lambda,
                    backbone_norm=cfg.backbone_norm,
                    pretrained_source=cfg.pretrained_source,
                    study_fusion=cfg.chex_fusion,
                    label_graph=cfg.label_graph,
                    label_graph_alpha=cfg.label_graph_alpha,
                    drop_path=cfg.drop_path)
    model = build_model(cfg.arch, pretrained=cfg.pretrained, **model_kw).to(device)
    eval_model = build_model(cfg.arch, pretrained=False, **model_kw).to(device)

    # AdamW：head 用 cfg.lr，backbone 用 cfg.lr*backbone_scale（预训练特征只做微调）；
    # bias 与 LayerNorm 不做 weight decay（这是 Transformer/ConvNeXt 上的通行做法）
    optimizer = torch.optim.AdamW(model.param_groups(
        cfg.lr, cfg.backbone_scale, cfg.weight_decay, cfg.layer_decay))
    ema = EMA(model, cfg.ema_decay)

    # 混合精度：T4 只能用 fp16（需 GradScaler），4090 用 bf16（不需要 scaler）
    ampf = amp_dtype(device) if cfg.amp else torch.float32
    use_scaler = ampf == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    # 冲突样本降权（可选）。conflicts.csv 是检查级的，先按 study_id 映射成
    # **全量 meta** 的图像级布尔数组，再用全局 row_idx 索引——这样与分折、
    # batch 顺序都无关。只在训练集上生效（验证集不降权，否则评估失真）。
    conf_mask = None
    if cfg.conflicts_csv:
        full_mask = load_conflict_rows(cfg.conflicts_csv, meta, labels)
        conf_mask = torch.from_numpy(full_mask)
        n_conf_img = int(full_mask[tr_sel].sum())
        n_conf_study = int(meta.loc[tr_sel, "Study_id"][
            full_mask[tr_sel]].nunique())
        if verbose:
            print(f"  冲突降权: {n_conf_study} 个检查 / {n_conf_img} 张训练图 "
                  f"→ 权重 {cfg.conflict_weight}", flush=True)

    # 按检查归一化权重：同样按**全量 meta** 建表、用全局 row_idx 索引。
    # 乘上 1/N 后每个检查对总损失的贡献恒等，与它有几张图无关。
    study_w = None
    if cfg.study_balanced:
        w_all = study_count_weights(meta["Study_id"].astype(str).to_numpy())
        study_w = torch.from_numpy(w_all)
        if verbose:
            n_tr = int(tr_sel.sum())
            mult = meta.loc[tr_sel, "Study_id"].astype(str).map(
                meta.loc[tr_sel, "Study_id"].astype(str).value_counts())
            print(f"  检查级归一化: {n_tr} 张训练图 / {int((~mult.isna()).sum())} 张，"
                  f"图片数 1~{int(mult.max())}（单图权重 "
                  f"{1.0 / int(mult.max()):.3f}~1.000）", flush=True)

    # 可选的正样本加权。默认关闭：指标是 AUC，过度加权会破坏稀有类的排序质量
    pos_weight = None
    if cfg.use_pos_weight:
        pos = lab_tr.sum(0).astype(np.float64)
        neg = len(lab_tr) - pos
        pw = np.clip((neg / np.maximum(pos, 1)) ** cfg.pos_weight_power,
                     1.0, cfg.pos_weight_cap)
        pos_weight = torch.tensor(pw, dtype=torch.float32, device=device)

    # LibAUC 在切换点之后使用自己的 PESG 学习率；BCE 阶段沿用完整
    # 训练长度的 cosine，避免在切换前就把 BCE 学习率衰减到接近 0。
    auc_loss_fn = None
    auc_optimizer = None
    auc_stage = False

    # 学习率调度的总步数与 warmup 步数（全部以**优化步**计，不是 epoch）
    steps_per_epoch = max(1, len(dl_tr))
    bce_epochs = (cfg.auc_start_epoch if cfg.loss == "auc"
                  else cfg.epochs)
    bce_epochs = max(1, min(cfg.epochs, bce_epochs))
    total_steps = steps_per_epoch * cfg.epochs
    warmup = steps_per_epoch * cfg.warmup_epochs
    base_lrs = [g["lr"] for g in optimizer.param_groups]   # 各组原始 lr，调度时乘倍率

    best = {"auc": -np.inf, "epoch": -1, "per_class": {}, "state": None,
            "probs": None, "row_idx": None}
    history = []
    step = 0

    for epoch in range(cfg.epochs):
        if cfg.loss == "auc" and epoch == bce_epochs:
            try:
                from libauc.losses import MultiLabelAUCMLoss
                from libauc.optimizers import PESG
            except ImportError as exc:
                raise RuntimeError(
                    "--loss auc 需要安装官方 LibAUC：python -m pip install -U libauc"
                ) from exc
            imratio = np.asarray(lab_tr, dtype=np.float32).mean(axis=0)
            imratio = np.clip(imratio, 1e-4, 1.0 - 1e-4).tolist()
            auc_loss_fn = MultiLabelAUCMLoss(
                margin=cfg.auc_margin, version=cfg.auc_version,
                imratio=imratio, num_labels=labels.shape[1], device=device)
            # PESG 同时更新模型参数和 AUC 损失的 a/b/alpha 辅助变量。
            auc_optimizer = PESG(
                model.parameters(), loss_fn=auc_loss_fn, lr=cfg.auc_lr,
                mode="adam", weight_decay=cfg.auc_weight_decay,
                epoch_decay=cfg.auc_epoch_decay, verbose=False, device=device)
            # LibAUC 的 PESG 默认将 epoch-level reference 初始化为随机张量；
            # 对 BCE 预训练后的微调，这会引入无意义的随机回拉。以当前
            # BCE 模型参数作为 reference，epoch_decay 才起到稳定微调的作用。
            if auc_optimizer.model_ref is not None:
                auc_vars = list(model.parameters()) + [auc_loss_fn.a, auc_loss_fn.b]
                with torch.no_grad():
                    for ref, var in zip(auc_optimizer.model_ref, auc_vars):
                        ref.copy_(var.detach())
            auc_stage = True
            if verbose:
                print(f"  LibAUC stage: epoch {epoch}~{cfg.epochs - 1}, "
                      f"lr={cfg.auc_lr:g}, imratio="
                      f"{[round(x, 4) for x in imratio]}", flush=True)

        model.train()
        t0 = time.time()
        running = 0.0

        for batch in dl_tr:
            if not auc_stage:
                # 逐个优化步更新学习率：warmup 爬升 → cosine 退火
                scale = cosine_lr(step, total_steps, warmup, 1.0)
                for g, b in zip(optimizer.param_groups, base_lrs):
                    g["lr"] = b * scale

            # non_blocking：pin_memory 的异步拷贝，与计算重叠
            image = batch["image"].to(device, non_blocking=True)
            view = batch["view_idx"].to(device, non_blocking=True)
            orient = batch["orient_idx"].to(device, non_blocking=True)
            size = batch["size_feat"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)

            # 侧位降权：从**本 batch 自己的** view_idx 推导，与 shuffle 顺序无关
            w = sample_weights(view, cfg.lateral_weight)
            # 冲突样本降权：用 batch 自带的全局 row_idx 查表，
            # **绝不能按 offset 位置切片**——shuffle 后位置与样本不对应，
            # 会像早期的侧位权重那样把降权贴到错误样本上且不报错。
            if conf_mask is not None:
                is_conf = conf_mask[batch["row_idx"].numpy()].to(device)
                w = w * torch.where(is_conf, cfg.conflict_weight, 1.0)
            # 检查级归一化：同样用 row_idx 查表，使每个检查权重和为 1
            if study_w is not None and not (cfg.study_mil or cfg.chex_fusion):
                w = w * study_w[batch["row_idx"].numpy()].to(device)

            active_optimizer = auc_optimizer if auc_stage else optimizer
            active_optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=ampf, enabled=cfg.amp):
                if cfg.chex_fusion:
                    if auc_stage or cfg.loss != "bce":
                        raise ValueError("CheXFusion-lite 当前只支持 --loss bce")
                    study_idx = batch["study_idx"].to(device, non_blocking=True)
                    logits, first = model.forward_study(
                        image, view, orient, size, study_idx)
                    starts = torch.ones_like(study_idx, dtype=torch.bool)
                    if len(study_idx) > 1:
                        starts[1:] = study_idx[1:] != study_idx[:-1]
                    group = torch.cumsum(starts.to(torch.long), dim=0) - 1
                    counts = torch.bincount(group, minlength=int(first.numel()))
                    target = target[first]
                    w_sum = torch.zeros(len(first), device=device)
                    w_sum.index_add_(0, group, w)
                    w = w_sum / counts.to(device).clamp_min(1)
                else:
                    logits = model(image, view, orient, size)
                if cfg.study_mil:
                    if auc_stage:
                        raise ValueError("Study-MIL 当前只支持 BCE/ASL，不能与 --loss auc 同时使用")
                    logits, first, group, counts = aggregate_batch_studies(
                        logits, batch["study_idx"].to(device, non_blocking=True))
                    target = target[first]
                    w_sum = torch.zeros(len(first), device=device)
                    w_sum.index_add_(0, group, w)
                    w = w_sum / counts.to(device).clamp_min(1)
                if auc_stage:
                    # AUC surrogate 接收连续的排序分数。这里必须使用 raw
                    # logits：BCE 预训练后的 sigmoid 可能已经饱和，继续对
                    # sigmoid 求导会把 AUC 阶段的有效梯度压到接近 0。
                    # AUC 只依赖排序，raw logits 与 sigmoid 保持相同排序。
                    # AUC 阶段不使用图像权重，保持排序目标不被 BCE 权重改变。
                    loss = auc_loss_fn(logits.float(), target.float(), auto=False)
                elif cfg.loss == "asl":
                    loss = asl_multilabel(
                        logits, target, gamma_neg=cfg.asl_gamma_neg,
                        gamma_pos=cfg.asl_gamma_pos, clip=cfg.asl_clip,
                        pos_weight=pos_weight, sample_weight=w,
                        label_smoothing=cfg.label_smoothing)
                else:
                    loss = bce_multilabel(logits, target, pos_weight=pos_weight,
                                          sample_weight=w,
                                          label_smoothing=cfg.label_smoothing)
                    if teacher_all is not None:
                        teacher = torch.from_numpy(
                            teacher_all[batch["row_idx"].numpy()]).to(device)
                        soft_loss = distill_multilabel(
                            logits, teacher,
                            temperature=cfg.distill_temperature,
                            sample_weight=w)
                        loss = ((1.0 - cfg.distill_alpha) * loss
                                + cfg.distill_alpha * soft_loss)
                # 标签图细化：对耦合矩阵加 L1，促稀疏（学出少量强关联即可）
                if model.label_graph is not None and cfg.label_graph_l1 > 0:
                    loss = loss + l1_penalty(model.label_graph, cfg.label_graph_l1)

            if auc_stage:
                # PESG 自己管理模型参数和辅助变量，直接调用 step。
                loss.backward()
                auc_optimizer.step()
            elif use_scaler:
                # fp16 路径：必须先 scale 再 backward，否则梯度会下溢成 0
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)          # 裁剪前必须还原真实梯度
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()

            ema.update(model)
            running += float(loss.detach())
            step += 1

        if auc_stage:
            # 官方建议周期性更新 PESG 的 epoch-level regularizer；
            # 两阶段微调只用温和 decay，避免学习率骤降。
            auc_optimizer.update_regularizer(decay_factor=2)

        ema.copy_to(eval_model)
        if cfg.chex_fusion:
            study_probs, study_rows, img_probs, row_idx = evaluate_chex_fusion(
                eval_model, dl_va_fusion, device)
            auc, per_class = macro_auc(labels[study_rows], study_probs)
        else:
            img_probs, row_idx = evaluate(eval_model, dl_va, device)
            auc, per_class = study_level_auc(img_probs, row_idx, meta, labels)

        history.append({"epoch": epoch, "loss": running / steps_per_epoch,
                        "val_macro_auc": auc, "sec": round(time.time() - t0, 1)})
        if verbose:
            print(f"  fold{fold} ep{epoch:02d} loss {running / steps_per_epoch:.4f} "
                  f"macroAUC {auc:.4f}  ({time.time() - t0:.0f}s)", flush=True)

        # 至少保证有可保存的产物，即使所有 epoch 的 AUC 都是 nan（小样本冒烟时会发生）
        if best["probs"] is None:
            best.update(probs=img_probs.copy(), row_idx=row_idx.copy())
        if np.isfinite(auc) and auc > best["auc"]:
            best.update(auc=float(auc), epoch=epoch, per_class=per_class,
                        probs=img_probs.copy(), row_idx=row_idx.copy(),
                        state={k: v.detach().cpu().clone()
                               for k, v in eval_model.state_dict().items()})

    out_dir = os.path.join(cfg.runs_dir, cfg.exp_name, f"fold{fold}")
    os.makedirs(out_dir, exist_ok=True)

    state = best["state"] if best["state"] is not None else \
        {k: v.detach().cpu().clone() for k, v in eval_model.state_dict().items()}
    torch.save({"model": state, "arch": cfg.arch, "image_size": cfg.image_size,
                "cache_size": cfg.cache_size, "cfg": cfg.to_json(),
                "best_auc": best["auc"], "best_epoch": best["epoch"]},
               os.path.join(out_dir, "ckpt.pt"))

    if cfg.save_oof:
        r = best["row_idx"]
        # study_id 必须存成定长字符串数组：pandas 的 object dtype 会让 npz
        # 变成 object array，加载端必须开 allow_pickle=True，跨环境不可靠。
        # view_idx 一并存下，供后续验证"侧位图该不该参与检查级聚合"。
        np.savez_compressed(
            os.path.join(out_dir, "oof.npz"),
            probs=best["probs"],
            row_idx=r,
            study_id=np.asarray(meta["Study_id"].to_numpy()[r], dtype=str),
            view_idx=meta["view_idx"].to_numpy()[r],
            y=labels[r],
        )

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"fold": fold, "best_auc": best["auc"], "best_epoch": best["epoch"],
                   "per_class": {str(k): v for k, v in (best["per_class"] or {}).items()},
                   "history": history}, f, ensure_ascii=False, indent=2)

    return {"fold": fold, "auc": best["auc"], "epoch": best["epoch"]}


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--arch", default="convnext_tiny")
    p.add_argument("--image-size", type=int, default=384)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--folds", default="0")
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--limit-images", type=int, default=None)
    p.add_argument("--seed", type=int, default=0,
                   help="数据划分种子（决定 5 折怎么分）。做多 seed 集成时不要改它")
    p.add_argument("--init-seed", type=int, default=None,
                   help="模型初始化种子。多 seed 集成就固定 --seed、只改这个")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--pretrained-source", default="torchvision",
                   choices=["torchvision", "imagenet22k", "clip_laion", "ssl_cxr"],
                   help="预训练来源；imagenet22k / clip_laion 仅支持 ConvNeXt-Base；"
                        "ssl_cxr = tools/pretrain_simmim.py 在官方训练集上自监督的权重")
    p.add_argument("--ssl-weights", default=None,
                   help="--pretrained-source ssl_cxr 时的权重文件路径")
    p.add_argument("--cache-dir", default="cache", help="缓存目录，默认相对当前工作目录")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--cache-size", type=int, default=448)
    # 消融开关（每次只改一个，其余保持默认，结果才能归因）
    p.add_argument("--loss", default="bce", choices=["bce", "asl", "auc"],
                   help="bce=原方案；asl=非对称损失；auc=BCE预训练后LibAUC微调")
    p.add_argument("--auc-start-epoch", type=int, default=6,
                   help="--loss auc 时从第几轮切换到 MultiLabelAUCMLoss+PESG")
    p.add_argument("--auc-lr", type=float, default=1e-4,
                   help="LibAUC PESG 微调学习率")
    p.add_argument("--auc-margin", type=float, default=1.0,
                   help="MultiLabelAUCMLoss margin")
    p.add_argument("--label-smooth", type=float, default=0.0,
                   help="标签平滑系数(0~0.5)，针对标签噪声。0=原行为，常用 0.1")
    p.add_argument("--conflicts-csv", default=None,
                   help="find_conflicts.py --export 导出的冲突检查清单；"
                        "指定后对这些检查的图片降权（配合 --conflict-weight）")
    p.add_argument("--conflict-weight", type=float, default=0.3,
                   help="冲突样本的损失权重，默认 0.3（1.0 = 不降权）")
    p.add_argument("--study-balanced", action="store_true",
                   help="按检查归一化样本权重（每张图 1/该检查图片数），"
                        "使每个检查对总损失贡献恒等，对齐检查级评测口径")
    p.add_argument("--study-mil", action="store_true",
                   help="按 Study 完整分组训练；同一检查的图像先 mean pooling 再计算 BCE/ASL")
    p.add_argument("--chex-fusion", action="store_true",
                   help="启用合规 CheXFusion-lite：Study 内一层 Transformer 特征融合")
    p.add_argument("--distill-targets", default=None,
                   help="OOF 教师软目标 npz；只对普通图像级 BCE 学生生效")
    p.add_argument("--distill-alpha", type=float, default=0.3,
                   help="蒸馏损失权重，0=普通监督")
    p.add_argument("--distill-temperature", type=float, default=2.0,
                   help="多标签 sigmoid 蒸馏温度")
    p.add_argument("--pool", default="gap",
                   choices=["gap", "gem", "attention", "csra", "ml_decoder"],
                   help="池化方式；ml_decoder 为轻量类别 query 空间解码头")
    p.add_argument("--csra-lambda", type=float, default=0.5,
                   help="CSRA 最大空间响应的残差权重")
    p.add_argument("--label-graph", action="store_true",
                   help="启用标签图细化（显式建模标签共现）")
    p.add_argument("--backbone-norm", action="store_true",
                   help="保留 ConvNeXt/Swin 的预训练末端归一化")
    p.add_argument("--aug", default="normal", choices=["normal", "strong", "weak"],
                   help="normal=基线；strong=更强增强；weak=保留整幅胸廓的弱增强")
    p.add_argument("--no-clahe", action="store_true", help="关闭 CLAHE")
    p.add_argument("--lateral-weight", type=float, default=0.5,
                   help="侧位图损失权重；1.0=不降权，0.0=完全不用侧位图训练")
    p.add_argument("--use-pos-weight", action="store_true", help="启用逐类别正样本加权")
    p.add_argument("--drop-path", type=float, default=None,
                   help="覆盖 stochastic depth（convnext_base 默认 0.5，微调常用 0.1~0.2）")
    p.add_argument("--layer-decay", type=float, default=1.0,
                   help="ConvNeXt 逐层学习率衰减，1.0=关闭；常用 0.7~0.85")
    p.add_argument("--backbone-scale", type=float, default=0.3,
                   help="主干 lr = lr * backbone-scale（默认 0.3，与旧实验一致）")
    args = p.parse_args(argv)
    if args.study_mil and args.chex_fusion:
        p.error("--study-mil 与 --chex-fusion 只能选一个")
    if args.chex_fusion and args.loss != "bce":
        p.error("--chex-fusion 当前只支持 --loss bce")
    if args.distill_targets and (args.study_mil or args.chex_fusion):
        p.error("蒸馏学生当前只支持普通图像级训练，不能与 Study-MIL/CheXFusion 同时使用")
    if not 0.0 <= args.distill_alpha <= 1.0:
        p.error("--distill-alpha 必须在 0 和 1 之间")
    if args.pretrained_source == "ssl_cxr":
        if not args.ssl_weights:
            p.error("--pretrained-source ssl_cxr 需要 --ssl-weights")
        os.environ["CXR_SSL_WEIGHTS"] = os.path.abspath(args.ssl_weights)

    cfg = TrainConfig(
        arch=args.arch, image_size=args.image_size, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, num_workers=args.num_workers,
        limit_images=args.limit_images, seed=args.seed, device=args.device,
        pretrained=not args.no_pretrained, pretrained_source=args.pretrained_source,
        runs_dir=args.runs_dir,
        cache_dir=args.cache_dir, cache_size=args.cache_size,
        loss=args.loss, auc_start_epoch=args.auc_start_epoch, auc_lr=args.auc_lr,
        auc_margin=args.auc_margin, pool=args.pool, csra_lambda=args.csra_lambda,
        backbone_norm=args.backbone_norm,
        label_graph=args.label_graph,
        aug=args.aug, use_clahe=not args.no_clahe,
        lateral_weight=args.lateral_weight, use_pos_weight=args.use_pos_weight,
        label_smoothing=args.label_smooth, init_seed=args.init_seed,
        conflicts_csv=args.conflicts_csv, conflict_weight=args.conflict_weight,
        study_balanced=args.study_balanced, study_mil=args.study_mil,
        chex_fusion=args.chex_fusion,
        distill_targets=args.distill_targets,
        distill_alpha=args.distill_alpha,
        distill_temperature=args.distill_temperature,
        drop_path=args.drop_path, layer_decay=args.layer_decay,
        backbone_scale=args.backbone_scale,
        extra={"ssl_weights": os.environ.get("CXR_SSL_WEIGHTS")}
        if args.pretrained_source == "ssl_cxr" else {},
    )
    device = torch.device(cfg.device)
    print(f"device={device} arch={cfg.arch} size={cfg.image_size} epochs={cfg.epochs}")
    print(f"cache={cfg.cache_dir}  runs={cfg.runs_dir}")
    print(f"loss={cfg.loss} pool={cfg.pool} backbone_norm={cfg.backbone_norm} "
          f"label_graph={cfg.label_graph} "
          f"aug={cfg.aug} clahe={cfg.use_clahe} lateral_w={cfg.lateral_weight} "
          f"label_smooth={cfg.label_smoothing} init_seed={cfg.init_seed} "
          f"auc_start={cfg.auc_start_epoch} auc_lr={cfg.auc_lr:g} "
          f"study_mil={cfg.study_mil} chex_fusion={cfg.chex_fusion} "
          f"distill={bool(cfg.distill_targets)} drop_path={cfg.drop_path} "
          f"layer_decay={cfg.layer_decay} backbone_scale={cfg.backbone_scale}")
    print(f"实验目录: {os.path.join(cfg.runs_dir, cfg.exp_name)}")

    meta, labels, norm = load_cache_index(cfg)
    folds = group_kfold(fold_frame(meta, labels), cfg.num_splits, cfg.seed)

    results = []
    for fold in [int(x) for x in args.folds.split(",")]:
        print(f"\n=== fold {fold} === train={int((folds != fold).sum())} "
              f"val={int((folds == fold).sum())}")
        results.append(train_fold(cfg, fold, folds, meta, labels, norm, device))

    print("\n结果：")
    for r in results:
        print(f"  fold{r['fold']}  macroAUC {r['auc']:.4f} (ep{r['epoch']})")
    print(f"  平均 {np.mean([r['auc'] for r in results]):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
