"""DINOv2 ViT-B/14 骨干（不依赖 timm / torch.hub 代码）。

权重：Meta 官方 ``dinov2_vitb14_pretrain.pth``，Apache-2.0。
上游数据：LVD-142M（通用自然图像检索整理，非医学），见 DINOv2 论文 arXiv:2304.07193。

参数名与官方 checkpoint 一一对应（实测 175 个 key：cls_token / pos_embed /
mask_token / patch_embed.proj / blocks.{0..11}.{norm1,attn.qkv,attn.proj,ls1.gamma,
norm2,mlp.fc1,mlp.fc2,ls2.gamma} / norm），因此加载时 strict 校验主干参数全覆盖；
mask_token 只用于自监督预训练，这里丢弃。

输出布局与 Swin 一致：(B, H/14, W/14, C) 的 patch token，已过末端 LayerNorm，
CXRModel 按 "hwc" 布局做全局平均池化。cls token 参与注意力计算，但不输出。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

DINOV2_VITB14_URL = (
    "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth")


class _DropPath(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep


class _LayerScale(nn.Module):
    def __init__(self, dim: int, init: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class _PatchEmbed(nn.Module):
    def __init__(self, patch: int, in_chans: int, dim: int):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, dim, kernel_size=patch, stride=patch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)          # (B, C, H/p, W/p)


class _Attention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        # 官方实现的排列：(B, N, 3, heads, head_dim) → (3, B, heads, N, head_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
        return self.proj(x.transpose(1, 2).reshape(B, N, C))


class _Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class _Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float, drop_path: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = _Attention(dim, heads)
        self.ls1 = _LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = _Mlp(dim, int(dim * mlp_ratio))
        self.ls2 = _LayerScale(dim)
        self.drop_path = _DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.ls1(self.attn(self.norm1(x))))
        return x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))


class DinoV2ViT(nn.Module):
    """ViT-B/14，结构与官方 DINOv2 一致（无 register token）。"""

    def __init__(self, dim: int = 768, depth: int = 12, heads: int = 12,
                 mlp_ratio: float = 4.0, patch: int = 14, in_chans: int = 3,
                 base_grid: int = 37, drop_path: float = 0.1):
        super().__init__()
        self.patch = patch
        self.base_grid = base_grid
        self.patch_embed = _PatchEmbed(patch, in_chans, dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + base_grid * base_grid, dim))
        self.mask_token = nn.Parameter(torch.zeros(1, dim))   # 仅为对齐 checkpoint
        self.mask_token.requires_grad_(False)
        # stochastic depth 按深度线性递增（官方做法）
        rates = [drop_path * i / max(depth - 1, 1) for i in range(depth)]
        self.blocks = nn.ModuleList(
            [_Block(dim, heads, mlp_ratio, r) for r in rates])
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=1e-6)

    def _pos_embed(self, gh: int, gw: int, dtype) -> torch.Tensor:
        cls_pos, grid = self.pos_embed[:, :1], self.pos_embed[:, 1:]
        if gh == self.base_grid and gw == self.base_grid:
            return self.pos_embed.to(dtype)
        g = self.base_grid
        grid = grid.reshape(1, g, g, -1).permute(0, 3, 1, 2).float()
        grid = F.interpolate(grid, size=(gh, gw), mode="bicubic", align_corners=False)
        grid = grid.permute(0, 2, 3, 1).reshape(1, gh * gw, -1)
        return torch.cat([cls_pos.float(), grid], dim=1).to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]
        if H % self.patch or W % self.patch:
            raise ValueError(f"DINOv2 输入边长需为 {self.patch} 的倍数，得到 {H}x{W}")
        x = self.patch_embed(x)                       # (B, C, gh, gw)
        B, C, gh, gw = x.shape
        x = x.flatten(2).transpose(1, 2)              # (B, gh*gw, C)
        x = torch.cat([self.cls_token.expand(B, -1, -1).to(x.dtype), x], dim=1)
        x = x + self._pos_embed(gh, gw, x.dtype)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 1:].reshape(B, gh, gw, C)         # 丢 cls，输出 hwc


def load_dinov2_pretrained(model: DinoV2ViT, url: str = DINOV2_VITB14_URL) -> None:
    """加载官方权重；任一主干参数缺失或形状不符即报错（不静默跳过）。"""
    state = torch.hub.load_state_dict_from_url(
        url, map_location="cpu", check_hash=False, progress=True)
    state = state.get("model", state)
    target = model.state_dict()
    missing = [k for k in target if k not in state]
    unexpected = [k for k in state if k not in target]
    bad = [k for k in target if k in state and state[k].shape != target[k].shape]
    if missing or unexpected or bad:
        raise RuntimeError(
            f"DINOv2 权重不匹配：缺失 {missing[:5]}，多余 {unexpected[:5]}，形状不符 {bad[:5]}")
    model.load_state_dict(state, strict=True)


def layer_id(name: str, depth: int = 12) -> int:
    """LLRD 层号：patch_embed/cls/pos = 0，blocks.i = i+1，末端 norm = depth+1。"""
    if name.startswith(("patch_embed.", "cls_token", "pos_embed", "mask_token")):
        return 0
    if name.startswith("blocks."):
        return int(name.split(".")[1]) + 1
    return depth + 1
