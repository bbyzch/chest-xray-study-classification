"""损失与训练辅助：BCE、EMA、学习率调度、混合精度选择。

这里的每个工具都对应一个具体的训练决策，注释里说明了为什么这么选。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def smooth_targets(targets: torch.Tensor, smoothing: float) -> torch.Tensor:
    """将胸片多热标签平滑至 [s/2, 1-s/2]；s<=0 时仅转为浮点。"""
    if smoothing <= 0:
        return targets.float()
    return targets.float() * (1.0 - smoothing) + 0.5 * smoothing


def bce_multilabel(logits, targets, pos_weight=None, sample_weight=None,
                   label_smoothing: float = 0.0):
    """十类别BCE：先按类别平均，再应用图像级权重并对批次平均。

    pos_weight 仅加权阳性；sample_weight 用于侧位和检查级归一化，
    label_smoothing 为0时保持原始多热标签。"""
    smoothed_labels = smooth_targets(targets, label_smoothing)
    loss = F.binary_cross_entropy_with_logits(
        logits, smoothed_labels, reduction="none", pos_weight=pos_weight)
    loss = loss.mean(dim=1)                    # 先对 10 个类别求平均
    if sample_weight is not None:
        loss = loss * sample_weight            # 再对样本加权
    return loss.mean()


def distill_multilabel(logits, teacher_probs, temperature: float = 2.0,
                       sample_weight=None):
    """多标签监督蒸馏：对每个 sigmoid 类别蒸馏教师软目标。

    温度作用在教师/学生 logit 上；没有 softmax，也不会把十个独立类别
    错误地当作互斥分类。教师概率应来自训练集 OOF，避免同图泄漏。
    """
    if temperature <= 0:
        raise ValueError("蒸馏温度必须大于 0")
    teacher_probs = teacher_probs.float().clamp(1e-5, 1.0 - 1e-5)
    teacher_logits = torch.logit(teacher_probs) / temperature
    loss = F.binary_cross_entropy_with_logits(
        logits / temperature, torch.sigmoid(teacher_logits), reduction="none")
    loss = loss.mean(dim=1) * (temperature ** 2)
    if sample_weight is not None:
        loss = loss * sample_weight
    return loss.mean()


def asl_multilabel(logits, targets, gamma_neg: float = 4.0, gamma_pos: float = 0.0,
                   clip: float = 0.05, eps: float = 1e-8, pos_weight=None,
                   sample_weight=None, label_smoothing: float = 0.0):
    """非对称损失 ASL（Ben-Baruch et al., 2020）。

    为什么可能比 BCE 好：多标签医学数据集里**负样本占绝对多数**，
    一个简单负样本的梯度会让模型学到"全判负"。ASL 做两件事：
      - gamma_neg 大：容易的负样本被大幅降权
      - clip：把负样本概率下界抬到 clip（默认 0.05），硬负样本的梯度直接归零
    正样本侧保持温和（gamma_pos=0），因为漏诊的代价更高。

    与 pos_weight 可叠加：pos_weight 调正样本权重，gamma_neg 调负样本难度权重，
    两者作用在不同维度上。
    """
    targets = smooth_targets(targets, label_smoothing)
    positive_probs = torch.sigmoid(logits)
    finding_probs = positive_probs
    absent_probs = 1.0 - positive_probs
    if clip > 0:
        # 概率下界抬升：负样本概率低于 clip 的部分不再产生梯度（防简单负样本主导）
        absent_probs = (absent_probs + clip).clamp(max=1.0)

    positive_log_likelihood = targets * torch.log(finding_probs.clamp(min=eps))
    negative_log_likelihood = (1.0 - targets) * torch.log(absent_probs.clamp(min=eps))
    loss = positive_log_likelihood + negative_log_likelihood

    if gamma_pos > 0 or gamma_neg > 0:
        target_probs = finding_probs * targets + absent_probs * (1.0 - targets)
        focusing_power = gamma_pos * targets + gamma_neg * (1.0 - targets)
        loss = loss * torch.pow((1.0 - target_probs).clamp(min=eps), focusing_power)

    loss = -loss
    if pos_weight is not None:
        # 与 BCE 口径对齐：只放大正样本那一项
        loss = loss * (targets * pos_weight + (1.0 - targets))
    loss = loss.mean(dim=1)
    if sample_weight is not None:
        loss = loss * sample_weight
    return loss.mean()


def l1_penalty(module, coef: float):
    """对模块里的参数做 L1（用于标签图耦合矩阵的稀疏化）。"""
    if module is None or coef <= 0:
        return torch.zeros((), device=next(module.parameters()).device)
    total = None
    for p in module.parameters():
        s = p.abs().sum()
        total = s if total is None else total + s
    return coef * total


class EMA:
    """指数滑动平均。

    推理一律用 EMA 权重——对 CV 稳定性的提升几乎免费，是性价比最高的技巧之一。
    shadow 只记录浮点参数（不含 BN 的 num_batches_tracked 这类整型统计量）；
    显式转 float()，否则混合精度训练时 fp16 参数会在累积中丢精度。
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.module = model
        self.decay = decay
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items() if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        """shadow = shadow * decay + param * (1 - decay)"""
        for k, v in model.state_dict().items():
            if k not in self.shadow:
                continue
            self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)

    def copy_to(self, model: torch.nn.Module):
        """把 EMA 权重灌进另一个模型（用于推理），保持目标模型的 dtype。"""
        model_weights = model.state_dict()
        for k, v in self.shadow.items():
            if k in model_weights:
                model_weights[k] = v.to(model_weights[k].dtype)
        model.load_state_dict(model_weights)

    def state_dict(self):
        return self.shadow


def cosine_lr(step: int, total: int, warmup: int, base_lr: float,
              min_ratio: float = 0.02) -> float:
    """线性 warmup + cosine 退火，返回 lr 的**倍率**（相对 base_lr）。

    为什么要 warmup：训练初期 head 是随机初始化的，直接大学习率会把
    预训练的 backbone 特征冲坏。前 1 个 epoch 线性爬升能避免这个问题。
    """
    import math

    if step < warmup:
        return (step + 1) / max(1, warmup)      # 线性爬升到 1.0
    # warmup 之后按余弦从 1.0 退到 min_ratio
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


def amp_dtype(device: torch.device) -> torch.dtype:
    """按设备选择混合精度 dtype。

    **必须显式判断 bf16 支持**：国赛评测卡 T4 是 Turing(sm_75)，不支持 bf16，
    只能走 fp16（并配 GradScaler）。训练卡 4090 是 Ada，bf16 更快更稳。
    这条差异是训练环境与评测环境之间最容易翻车的点，见设计文档 §11.2。
    """
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.bfloat16        # CPU 上用 bf16 即可
