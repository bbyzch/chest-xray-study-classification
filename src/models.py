"""模型：torchvision 原生骨干 + 视位/体位/尺寸 旁路 embedding 头。

为什么用 torchvision 原生骨干（设计文档 §4.3 与风险 R3）：
国赛容器内置 torchvision 0.25.0，但**未必有 timm、未必联网**。
用原生骨干可以在容器里零额外依赖地构造模型。

输出约定：**(B, 10) logits**，索引 0..8 对应类别 1..9（病灶），
索引 9 对应类别 10（无异常）。融合发生在聚合阶段（aggregate.py），
训练期保持 10 个独立 sigmoid，便于消融。
"""
from __future__ import annotations

import torch
import torch.nn as nn


ARCHS = {
    "convnext_tiny": ("convnext", 768),
    "convnext_small": ("convnext", 768),
    "convnext_base": ("convnext", 1024),
    "efficientnet_b0": ("efficientnet", 1280),
    "efficientnet_b1": ("efficientnet", 1280),
    "efficientnet_b3": ("efficientnet", 1536),
    "swin_v2_t": ("swin", 768),
    "swin_v2_s": ("swin", 768),
    "resnet50": ("resnet", 2048),
    # DINOv2 自监督 ViT-B/14（Apache-2.0，LVD-142M 自然图像）；输入边长须为 14 的倍数
    "dinov2_vitb14": ("dinov2", 768),
}

CONVNEXT_22K_URL = (
    "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_1k_224.pth"
)

# CLIP（LAION-2B 图文对比预训练）→ ImageNet-12K → ImageNet-1K@384 微调的 ConvNeXt-Base，
# timm 发布，Apache-2.0。结构与 torchvision convnext_base 完全相同（[3,3,27,3]，128→1024）。
# 规则第 1 条：LAION 属于通用公开数据，合规。
CONVNEXT_CLIP_REPO = "timm/convnext_base.clip_laion2b_augreg_ft_in12k_in1k_384"
CONVNEXT_CLIP_FILE = "convnext_base.clip_laion2b_augreg_ft_in12k_in1k_384.safetensors"
PRETRAINED_SOURCES = ("torchvision", "imagenet22k", "clip_laion", "ssl_cxr")


def _to_1ch_conv(conv: nn.Conv2d) -> nn.Conv2d:
    """把 3 通道卷积改成 1 通道，权重沿通道维**求和**。

    X 光片是单通道灰度。若把灰度值重复成 3 通道再前向，等价于用
    ``weight.sum(dim=1)`` 的单通道卷积——所以求和能精确保留预训练特征，
    而不是丢掉 2/3 的权重（取均值会让响应缩小 3 倍）。
    """
    gray_conv = nn.Conv2d(
        1, conv.out_channels, conv.kernel_size,
        stride=conv.stride, padding=conv.padding, dilation=conv.dilation,
        groups=1, bias=conv.bias is not None, padding_mode=conv.padding_mode,
    )
    with torch.no_grad():
        gray_conv.weight.copy_(conv.weight.sum(dim=1, keepdim=True))
        if conv.bias is not None:
            gray_conv.bias.copy_(conv.bias)
    return gray_conv


def _adapt_first_conv(module: nn.Module) -> bool:
    """深度优先找到第一个 3 通道 Conv2d 并替换为单通道版本。"""
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d) and child.in_channels == 3:
            setattr(module, name, _to_1ch_conv(child))
            return True
        if _adapt_first_conv(child):
            return True
    return False


def _official_convnext_key(key: str) -> str | None:
    """Map an official ConvNeXt checkpoint key to torchvision's key layout."""
    if key.startswith("module."):
        key = key[7:]
    if key.startswith("model."):
        key = key[6:]
    if key.startswith("downsample_layers."):
        rest = key[len("downsample_layers."):]
        stage, suffix = rest.split(".", 1)
        return f"features.{2 * int(stage)}.{suffix}"
    if key.startswith("stages."):
        rest = key[len("stages."):]
        stage, block, parameter = rest.split(".", 2)
        if "." in parameter:
            name, suffix = parameter.split(".", 1)
        else:
            name, suffix = parameter, ""
        block_map = {
            "dwconv": "block.0",
            "norm": "block.2",
            "pwconv1": "block.3",
            "pwconv2": "block.5",
            "gamma": "layer_scale",
        }
        mapped = block_map.get(name)
        if mapped is None:
            return None
        target = f"features.{2 * int(stage) + 1}.{block}.{mapped}"
        return target if not suffix else f"{target}.{suffix}"
    if key in ("norm.weight", "norm.bias"):
        return "classifier.0." + key.split(".", 1)[1]
    return None


def _load_official_convnext22k(backbone_model: nn.Module,
                               url: str = CONVNEXT_22K_URL):
    """Load the official ConvNeXt-B 22K->1K checkpoint without timm."""
    checkpoint = torch.hub.load_state_dict_from_url(
        url, map_location="cpu", check_hash=False, progress=True)
    source = checkpoint.get("model", checkpoint)
    target = backbone_model.state_dict()
    mapped = {}
    for source_key, value in source.items():
        target_key = _official_convnext_key(source_key)
        if target_key is None or target_key not in target:
            continue
        if target[target_key].shape != value.shape:
            # Official gamma is (C,), torchvision layer_scale is (C,1,1).
            if (target_key.endswith("layer_scale") and
                    value.shape == target[target_key].flatten().shape):
                value = value.reshape_as(target[target_key])
            else:
                raise RuntimeError(
                    f"ImageNet-22K 权重形状不匹配: {source_key} -> {target_key} "
                    f"{tuple(value.shape)} != {tuple(target[target_key].shape)}")
        mapped[target_key] = value

    required = [k for k in target if k.startswith("features.")]
    missing = [k for k in required if k not in mapped]
    if missing:
        raise RuntimeError(
            f"ImageNet-22K 权重映射不完整，缺少 {len(missing)} 个主干参数，"
            f"示例: {missing[:5]}")
    backbone_model.load_state_dict(mapped, strict=False)


_ST_DTYPES = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16,
              "F64": torch.float64, "I64": torch.int64, "I32": torch.int32}


def read_safetensors(path: str) -> dict:
    """纯 torch 读 .safetensors（格式：8 字节小端头长度 + JSON 头 + 原始字节）。

    不依赖 safetensors 包——评测容器未必装了它。
    """
    import json
    import struct
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n).decode("utf-8"))
        raw = bytearray(f.read())
    header.pop("__metadata__", None)
    out = {}
    for name, info in header.items():
        dtype = _ST_DTYPES.get(info["dtype"])
        if dtype is None:
            raise ValueError(f"{path}: 不支持的 dtype {info['dtype']}（{name}）")
        start, end = info["data_offsets"]
        t = torch.frombuffer(raw, dtype=dtype, count=(end - start) // torch.empty(
            0, dtype=dtype).element_size(), offset=start) if end > start else torch.empty(0, dtype=dtype)
        out[name] = t.reshape(info["shape"]).clone()
    return out


def _timm_convnext_key(key: str) -> str | None:
    """timm ConvNeXt 的参数名 → torchvision convnext 的参数名；分类层 head.fc 返回 None。"""
    if key.startswith("stem."):                       # stem.0 conv / stem.1 LayerNorm2d
        return "features.0." + key[len("stem."):]
    if key.startswith("head.norm."):                  # 末端 LayerNorm → classifier[0]
        return "classifier.0." + key[len("head.norm."):]
    if not key.startswith("stages."):
        return None
    parts = key.split(".")
    stage = int(parts[1])
    if parts[2] == "downsample":                      # stages.s.downsample.{0 norm,1 conv}
        return f"features.{2 * stage}." + ".".join(parts[3:])
    if parts[2] != "blocks":
        return None
    block, name = parts[3], parts[4]
    suffix = ".".join(parts[5:])
    mapped = {"conv_dw": "block.0", "norm": "block.2", "gamma": "layer_scale"}.get(name)
    if name == "mlp":                                 # mlp.fc1 / mlp.fc2
        mapped = {"fc1": "block.3", "fc2": "block.5"}[parts[5]]
        suffix = ".".join(parts[6:])
    if mapped is None:
        return None
    target = f"features.{2 * stage + 1}.{block}.{mapped}"
    return f"{target}.{suffix}" if suffix else target


def _clip_weights_path() -> str:
    """返回 CLIP ConvNeXt 权重的本地路径，没有就下载到 torch hub 缓存。

    CXR_CLIP_WEIGHTS 可直接指向手动下载好的文件；HF_ENDPOINT 默认 hf-mirror.com
    （国内访问不了 huggingface.co）。
    """
    import os
    local = os.environ.get("CXR_CLIP_WEIGHTS")
    if local:
        return local
    cache = os.path.join(torch.hub.get_dir(), "checkpoints", CONVNEXT_CLIP_FILE)
    if not os.path.isfile(cache):
        endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com").rstrip("/")
        url = f"{endpoint}/{CONVNEXT_CLIP_REPO}/resolve/main/model.safetensors"
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        print(f"Downloading: {url} -> {cache}")
        torch.hub.download_url_to_file(url, cache, progress=True)
    return cache


def load_timm_convnext(backbone_model: nn.Module, state: dict, label: str) -> None:
    """把 timm 格式的 ConvNeXt state_dict 严格映射进 torchvision 模型；缺任何主干参数都报错。"""
    target = backbone_model.state_dict()
    mapped = {}
    for source_key, value in state.items():
        target_key = _timm_convnext_key(source_key)
        if target_key is None:
            continue
        if target_key not in target:
            raise RuntimeError(f"{label}: {source_key} 映射到不存在的 {target_key}")
        if target[target_key].shape != value.shape:
            if (target_key.endswith("layer_scale")
                    and value.numel() == target[target_key].numel()):
                value = value.reshape_as(target[target_key])
            else:
                raise RuntimeError(
                    f"{label}: 形状不匹配 {source_key} -> {target_key} "
                    f"{tuple(value.shape)} != {tuple(target[target_key].shape)}")
        mapped[target_key] = value.to(target[target_key].dtype)
    required = [k for k in target if k.startswith("features.") or k.startswith("classifier.0.")]
    missing = [k for k in required if k not in mapped]
    if missing:
        raise RuntimeError(f"{label}: 映射不完整，缺少 {len(missing)} 个参数，示例 {missing[:5]}")
    backbone_model.load_state_dict(mapped, strict=False)


def _build_body(arch: str, pretrained: bool, backbone_norm: bool = False,
                pretrained_source: str = "torchvision",
                drop_path: float | None = None):
    """返回 body、特征维度、布局和可选预训练 norm；默认结构兼容旧权重。

    drop_path：覆盖 torchvision 的 stochastic_depth_prob。None 保持 torchvision
    默认值。注意 convnext_base/large 的默认值是 0.5，那是 ImageNet **从头训练**
    的配方；官方 ConvNeXt 的下游微调配方用 0.1~0.2。它不改变参数结构，
    推理（eval 模式）下 stochastic depth 不生效，所以旧 ckpt 不受影响。
    """
    if arch not in ARCHS:
        raise ValueError(f"不支持的骨干 {arch!r}，可选: {sorted(ARCHS)}")
    family, dim = ARCHS[arch]

    import torchvision.models as tvm

    if pretrained_source not in PRETRAINED_SOURCES:
        raise ValueError(f"未知预训练来源 {pretrained_source!r}，可选 {PRETRAINED_SOURCES}")
    if pretrained_source == "ssl_cxr" and family != "convnext":
        raise ValueError("ssl_cxr 目前只支持 ConvNeXt")
    if pretrained_source in ("imagenet22k", "clip_laion") and arch != "convnext_base":
        raise ValueError(f"{pretrained_source} 预训练目前只支持 convnext_base")
    weights = "DEFAULT" if pretrained and pretrained_source == "torchvision" else None
    sd_kw = {}
    if drop_path is not None:
        if family not in ("convnext", "swin", "dinov2"):
            raise ValueError("drop_path 仅适用于 ConvNeXt / Swin / DINOv2")
        if family != "dinov2":
            sd_kw["stochastic_depth_prob"] = float(drop_path)

    feature_norm = nn.Identity()
    if family == "dinov2":
        from src.dinov2 import DinoV2ViT, load_dinov2_pretrained
        if pretrained_source != "torchvision":
            raise ValueError("DINOv2 使用自身官方权重，不要再指定 --pretrained-source")
        body = DinoV2ViT(drop_path=0.1 if drop_path is None else float(drop_path))
        if pretrained:
            load_dinov2_pretrained(body)
        feat = dim
    elif family == "convnext":
        backbone_model = getattr(tvm, arch)(weights=weights, **sd_kw)
        if pretrained and pretrained_source == "imagenet22k":
            _load_official_convnext22k(backbone_model)
        elif pretrained and pretrained_source == "clip_laion":
            load_timm_convnext(backbone_model, read_safetensors(_clip_weights_path()),
                               "CLIP-LAION ConvNeXt")
        body, feat = backbone_model.features, dim
        if backbone_norm:
            feature_norm = backbone_model.classifier[0]
    elif family == "efficientnet":
        backbone_model = getattr(tvm, arch)(weights=weights)
        body, feat = backbone_model.features, backbone_model.classifier[1].in_features
    elif family == "swin":
        backbone_model = getattr(tvm, arch)(weights=weights, **sd_kw)
        body, feat = backbone_model.features, backbone_model.head.in_features
        if backbone_norm:
            feature_norm = backbone_model.norm
    elif family == "resnet":
        backbone_model = getattr(tvm, arch)(weights=weights)
        body = nn.Sequential(
            backbone_model.conv1, backbone_model.bn1, backbone_model.relu, backbone_model.maxpool,
            backbone_model.layer1, backbone_model.layer2, backbone_model.layer3, backbone_model.layer4,
        )
        feat = backbone_model.fc.in_features
    else:
        raise ValueError(family)

    if not _adapt_first_conv(body):
        raise RuntimeError(f"{arch}: 没找到可改造的首层卷积，单通道输入会失败")
    if backbone_norm and family not in ("convnext", "swin"):
        raise ValueError("backbone_norm 仅适用于 ConvNeXt / Swin")
    if pretrained and pretrained_source == "ssl_cxr":
        load_ssl_cxr(body, feature_norm, arch)
    return body, feat, ("hwc" if family in ("swin", "dinov2") else "chw"), feature_norm


def ssl_weights_path() -> str:
    """官方训练集自监督预训练权重（tools/pretrain_simmim.py 的产物）的路径。"""
    import os
    path = os.environ.get("CXR_SSL_WEIGHTS")
    if not path:
        raise RuntimeError("--pretrained-source ssl_cxr 需要 --ssl-weights（或环境变量 CXR_SSL_WEIGHTS）")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"找不到自监督权重: {path}")
    return path


def load_ssl_cxr(body: nn.Module, feature_norm: nn.Module, arch: str) -> None:
    """载入 pretrain_simmim.py 保存的主干（已是 1 通道首层）与末端 norm，逐键严格匹配。"""
    ck = torch.load(ssl_weights_path(), map_location="cpu", weights_only=True)
    if ck.get("format") != "cxr_ssl_v1":
        raise RuntimeError("不是 pretrain_simmim.py 产出的自监督权重")
    if ck.get("arch") != arch:
        raise RuntimeError(f"自监督权重的骨干是 {ck.get('arch')}，当前是 {arch}")
    body.load_state_dict(ck["body"], strict=True)
    if not isinstance(feature_norm, nn.Identity):
        feature_norm.load_state_dict(ck["feature_norm"], strict=True)


class LabelGraph(nn.Module):
    """标签图细化：参考 arXiv:2511.07801 的稀疏标签耦合方法。

    学习零对角耦合矩阵，以 sigmoid(logits) 的消息修正各病灶logits。
    矩阵零初始化保持初始预测；10类配置下矩阵含100个存储参数。
    方法及公开实现来源见 THIRD_PARTY_NOTICES.md。"""

    def __init__(self, num_classes: int = 10, alpha: float = 0.3):
        super().__init__()
        self.A = nn.Parameter(torch.zeros(num_classes, num_classes))
        self.alpha = alpha
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        diagonal_mask = torch.eye(self.num_classes, device=logits.device, dtype=self.A.dtype)
        off_diagonal_couplings = self.A * (1.0 - diagonal_mask)                      # 强制对角为 0
        return logits + self.alpha * (torch.sigmoid(logits) @ off_diagonal_couplings)


class GeMPool(nn.Module):
    """广义均值池化：p=1 退化为平均，p→∞ 退化为最大。

    CXR 的病灶多为**局部**征象（气胸在肺尖、积液在肋膈角），
    全局平均会把它们冲淡；GeM 让模型自己学 p，介于平均与最大之间。
    """

    def __init__(self, p: float = 3.0):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # (B,C,H,W)
        p = self.p.clamp(min=1.0)
        x = x.clamp(min=1e-6).pow(p)
        x = nn.functional.adaptive_avg_pool2d(x, 1).pow(1.0 / p)
        return torch.flatten(x, 1)


class AttentionPool(nn.Module):
    """注意力池化：给每个空间位置学一个权重，而不是一视同仁地平均。"""

    def __init__(self, channels: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1), nn.GELU(),
            nn.Conv2d(channels // 4, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # (B,C,H,W)
        spatial_attention = torch.softmax(self.score(x).flatten(2), dim=-1)  # (B,1,H*W)
        return (x.flatten(2) * spatial_attention).sum(-1)


class CSRAHead(nn.Module):
    """Class-specific residual attention logits over a spatial feature map."""

    def __init__(self, channels: int, side_dim: int, num_classes: int,
                 lam: float = 0.5):
        super().__init__()
        self.classifier = nn.Conv2d(channels, num_classes, kernel_size=1)
        self.side = nn.Linear(side_dim, num_classes)
        self.lam = float(lam)
        nn.init.zeros_(self.classifier.bias)
        nn.init.zeros_(self.side.bias)

    def forward(self, features: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
        raw = self.classifier(features).flatten(2)  # (B, classes, HW)
        image_logits = raw.mean(dim=-1) + self.lam * raw.amax(dim=-1)
        return image_logits + self.side(side)


class MLDecoderHead(nn.Module):
    """轻量 ML-Decoder 风格多标签头。

    每个类别拥有一个 query，通过一次 cross-attention 从空间特征 token
    中读取类别相关区域；输出仍是普通的 (B, num_classes) logits，因此
    现有 BCE、OOF、聚合和推理流程都可以直接复用。
    """

    def __init__(self, channels: int, side_dim: int, num_classes: int,
                 decoder_dim: int = 256, num_heads: int = 8):
        super().__init__()
        if decoder_dim % num_heads != 0:
            raise ValueError("MLDecoder decoder_dim 必须能被 num_heads 整除")
        self.proj = nn.Conv2d(channels, decoder_dim, kernel_size=1, bias=False)
        self.token_norm = nn.LayerNorm(decoder_dim)
        self.query = nn.Parameter(torch.randn(num_classes, decoder_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            decoder_dim, num_heads, dropout=0.1, batch_first=True)
        self.query_norm = nn.LayerNorm(decoder_dim)
        self.ffn = nn.Sequential(
            nn.Linear(decoder_dim, decoder_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(decoder_dim * 2, decoder_dim),
        )
        self.out = nn.Linear(decoder_dim, 1)
        self.side = nn.Linear(side_dim, num_classes)
        # Decoder 初始只提供零残差，避免随机 decoder 破坏已有基线预测。
        # 训练中 out/side 仍会获得梯度，随后逐步学习空间修正。
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        nn.init.zeros_(self.side.weight)
        nn.init.zeros_(self.side.bias)

    def forward(self, features: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
        tokens = self.proj(features).flatten(2).transpose(1, 2)
        tokens = self.token_norm(tokens)
        queries = self.query.unsqueeze(0).expand(features.shape[0], -1, -1)
        attended, _ = self.cross_attn(queries, tokens, tokens, need_weights=False)
        attended = self.query_norm(attended + queries)
        attended = attended + self.ffn(attended)
        logits = self.out(attended).squeeze(-1)
        return logits + self.side(side)


class CXRModel(nn.Module):
    def __init__(self, arch: str = "convnext_tiny", num_view: int = 6,
                 num_orient: int = 3, num_classes: int = 10,
                 pretrained: bool = True, view_dim: int = 8,
                 orient_dim: int = 4, hidden: int = 256, dropout: float = 0.3,
                 pool: str = "gap", label_graph: bool = False,
                 label_graph_alpha: float = 0.3, backbone_norm: bool = False,
                 csra_lambda: float = 0.5,
                 pretrained_source: str = "torchvision",
                 study_fusion: bool = False,
                 drop_path: float | None = None):
        super().__init__()
        # body 只到全局池化之前；分类头完全由我们自己定义
        self.body, feat_dim, self.layout, self.feature_norm = _build_body(
            arch, pretrained, backbone_norm, pretrained_source, drop_path)
        self.arch = arch
        self.pool_mode = pool
        self.study_fusion = bool(study_fusion)
        if self.study_fusion and pool != "gap":
            raise ValueError("CheXFusion-lite 当前要求 --pool gap")

        # 池化方式：gap 是原方案；gem / attention 针对"局部病灶被平均冲淡"的问题
        if pool == "gem":
            self.pool = GeMPool()
        elif pool == "attention":
            self.pool = AttentionPool(feat_dim)
        elif pool in ("csra", "ml_decoder"):
            self.pool = None
        elif pool == "gap":
            self.pool = None
        else:
            raise ValueError(f"未知池化方式 {pool!r}，可选 gap/gem/attention/csra/ml_decoder")

        # 旁路特征：视位与体位是**类别型**，用 embedding 学；
        # 它们携带强烈的先验（AP 的无异常率 22.7%，LATERAL 高达 62.2%），
        # 让模型能区分"这张图本来就难看出病灶"和"模型没学好"。
        self.view_emb = nn.Embedding(num_view, view_dim)
        self.orient_emb = nn.Embedding(num_orient, orient_dim)
        side_dim = view_dim + orient_dim + 2  # +2 给 log(Rows), log(Columns)

        if pool == "csra":
            self.head = None
            self.csra_head = CSRAHead(feat_dim, side_dim, num_classes, csra_lambda)
            self.ml_decoder_head = None
        elif pool == "ml_decoder":
            # 保留原始 GAP+MLP 路径作为强基线，ML-Decoder 只学习空间残差。
            self.head = nn.Sequential(
                nn.LayerNorm(feat_dim + side_dim),
                nn.Linear(feat_dim + side_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, num_classes),
            )
            nn.init.zeros_(self.head[-1].bias)
            self.csra_head = None
            self.ml_decoder_head = MLDecoderHead(feat_dim, side_dim, num_classes)
        else:
            # 头：LayerNorm 稳定不同骨干的池化特征尺度，两层 MLP 足够（数据只有 3 万张，
            # 头太深会过拟合）
            self.head = nn.Sequential(
                nn.LayerNorm(feat_dim + side_dim),
                nn.Linear(feat_dim + side_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, num_classes),
            )
            # 输出层 bias 置零：初始时 10 类概率都接近 0.5，起点不偏袒任何一类
            nn.init.zeros_(self.head[-1].bias)
            self.csra_head = None
            self.ml_decoder_head = None

        if self.study_fusion:
            fusion_dim = 256
            side_dim = view_dim + orient_dim + 2
            self.fusion_proj = nn.Linear(feat_dim + side_dim, fusion_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=fusion_dim, nhead=4, dim_feedforward=512,
                dropout=0.1, batch_first=True, norm_first=False,
            )
            self.fusion_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
            self.fusion_query = nn.Parameter(torch.zeros(1, 1, fusion_dim))
            self.fusion_head = nn.Sequential(
                nn.LayerNorm(fusion_dim),
                nn.Linear(fusion_dim, num_classes),
            )
            # 以零残差开始，保证初始 Study 预测与现有 mean 聚合一致。
            nn.init.zeros_(self.fusion_head[-1].weight)
            nn.init.zeros_(self.fusion_head[-1].bias)

        # 标签图细化（可选）。放在 head 之后、sigmoid 之前
        self.label_graph = (LabelGraph(num_classes, label_graph_alpha)
                            if label_graph else None)

    def forward(self, image, view_idx, orient_idx, size_feat) -> torch.Tensor:
        """返回 (B, 10) logits。索引 0..8 = 类别 1..9，索引 9 = 类别 10（无异常）。"""
        image_features = self.body(image)
        # Swin 的特征是 (B,H,W,C)，其余骨干是 (B,C,H,W)，统一成后者
        if self.layout == "hwc":
            # Swin 的预训练 norm 在每个 token 上、空间池化之前执行。
            image_features = self.feature_norm(image_features)
            image_features = image_features.permute(0, 3, 1, 2).contiguous()
        # 旁路特征拼到池化特征后面（ML-Decoder/CSRA 也使用这些采集信息）。
        acquisition_features = torch.cat([
            self.view_emb(view_idx),
            self.orient_emb(orient_idx),
            size_feat.to(image_features.dtype),
        ], dim=1)
        # CSRA / ML-Decoder 在空间特征图上直接生成每类 logits；其余模式先池化。
        if self.pool_mode == "ml_decoder":
            # ML-Decoder 读取空间 token；ConvNeXt 的 LayerNorm 需要先转成
            # channel-last，Swin 已在上方完成 norm 和布局统一。
            if self.layout == "chw" and not isinstance(self.feature_norm, nn.Identity):
                image_features = self.feature_norm(
                    image_features.permute(0, 2, 3, 1)
                ).permute(0, 3, 1, 2).contiguous()
            decoder_features = image_features
            pooled_features = nn.functional.adaptive_avg_pool2d(
                image_features, 1).flatten(1)
            base_logits = self.head(torch.cat(
                [pooled_features, acquisition_features], dim=1))
            decoder_logits = self.ml_decoder_head(
                decoder_features, acquisition_features)
            logits = base_logits + decoder_logits
        elif self.pool_mode == "csra":
            if self.layout == "chw" and not isinstance(self.feature_norm, nn.Identity):
                image_features = self.feature_norm(
                    image_features.permute(0, 2, 3, 1)
                ).permute(0, 3, 1, 2).contiguous()
        elif self.pool is None:
            image_features = nn.functional.adaptive_avg_pool2d(image_features, 1)
            if self.layout == "chw":
                image_features = self.feature_norm(image_features)
            image_features = torch.flatten(image_features, 1)
        else:
            image_features = self.pool(image_features)
            if self.layout == "chw":
                image_features = self.feature_norm(image_features[:, :, None, None]).flatten(1)

        if self.pool_mode == "ml_decoder":
            pass  # logits 已由 GAP 基线 + 空间 decoder 残差生成
        elif self.pool_mode == "csra":
            logits = self.csra_head(image_features, acquisition_features)
        else:
            logits = self.head(torch.cat([image_features, acquisition_features], dim=1))
        # 标签图细化：用 σ(z)·A 学到的共现结构修正 logits（可选）
        if self.label_graph is not None:
            logits = self.label_graph(logits)
        return logits

    def forward_study(self, image, view_idx, orient_idx, size_feat,
                      study_idx: torch.Tensor):
        """CheXFusion-lite：按连续排列的 Study 融合多张图并输出 Study logits。

        返回 ``(study_logits, first_positions)``，后者用于从重复的图像标签
        中取每个 Study 的一份标签，也便于保存兼容现有 OOF 的图像预测。
        """
        if not self.study_fusion:
            raise RuntimeError("模型未启用 study_fusion")
        if self.pool_mode != "gap":
            raise RuntimeError("study_fusion 只支持 gap 池化")
        image_features = self.body(image)
        if self.layout == "hwc":
            image_features = self.feature_norm(image_features)
            image_features = image_features.permute(0, 3, 1, 2).contiguous()
        if self.layout == "chw" and not isinstance(self.feature_norm, nn.Identity):
            image_features = self.feature_norm(
                image_features.permute(0, 2, 3, 1)
            ).permute(0, 3, 1, 2).contiguous()
        pooled = nn.functional.adaptive_avg_pool2d(image_features, 1).flatten(1)
        acquisition = torch.cat([
            self.view_emb(view_idx),
            self.orient_emb(orient_idx),
            size_feat.to(pooled.dtype),
        ], dim=1)
        image_logits = self.head(torch.cat([pooled, acquisition], dim=1))

        starts = torch.ones_like(study_idx, dtype=torch.bool)
        if len(study_idx) > 1:
            starts[1:] = study_idx[1:] != study_idx[:-1]
        first = torch.nonzero(starts, as_tuple=False).flatten()
        group = torch.cumsum(starts.to(torch.long), dim=0) - 1
        counts = torch.bincount(group, minlength=len(first))
        n_studies = len(first)
        max_images = int(counts.max().item())
        tokens = self.fusion_proj(torch.cat([pooled, acquisition], dim=1))
        padded = tokens.new_zeros((n_studies, max_images, tokens.shape[1]))
        padding = torch.ones((n_studies, max_images), dtype=torch.bool,
                            device=tokens.device)
        for i in range(n_studies):
            selected = group == i
            n = int(selected.sum().item())
            padded[i, :n] = tokens[selected]
            padding[i, :n] = False
        query = self.fusion_query.expand(n_studies, -1, -1)
        sequence = torch.cat([query, padded], dim=1)
        key_padding_mask = torch.cat([
            torch.zeros((n_studies, 1), dtype=torch.bool, device=tokens.device),
            padding,
        ], dim=1)
        fused = self.fusion_encoder(
            sequence, src_key_padding_mask=key_padding_mask)[:, 0]
        residual = self.fusion_head(fused)

        probs = torch.sigmoid(image_logits.float())
        sums = torch.zeros((n_studies, image_logits.shape[1]), device=image_logits.device)
        sums.index_add_(0, group, probs)
        base_probs = sums / counts.to(sums.device)[:, None].clamp_min(1)
        base_logits = torch.logit(base_probs.clamp(1e-5, 1.0 - 1e-5))
        return base_logits + residual, first

    def _backbone_layer_id(self, name: str) -> int:
        """ConvNeXt 主干参数的层号（0=stem … 7=最后一个 stage 与末端 norm），用于 LLRD。

        torchvision ConvNeXt 的 features：0=stem，1/3/5/7=四个 stage，
        2/4/6=下采样层。stage3（features.5，Base 有 27 个 block）按 block 号
        再三等分，避免最大的 stage 只占一个衰减档。
        """
        if name.startswith("feature_norm."):
            return 7
        parts = name.split(".")
        if len(parts) < 2 or parts[0] != "body" or not parts[1].isdigit():
            return 7
        idx = int(parts[1])
        if idx == 0:
            return 0
        if idx in (1, 2):
            return 1
        if idx in (3, 4):
            return 2
        if idx == 5:
            n_blocks = len(self.body[5])
            block = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
            return 3 + min(2, block * 3 // max(n_blocks, 1))
        return 6 if idx == 6 else 7  # features.6 下采样；features.7 最后一个 stage

    def param_groups(self, head_lr: float, backbone_scale: float = 0.3,
                     weight_decay: float = 0.05, layer_decay: float = 1.0):
        """head 用 head_lr，backbone 用 head_lr*backbone_scale；bias/LayerNorm 不做 weight decay。

        layer_decay < 1 时对主干做逐层学习率衰减（LLRD）：
        层号 i 的 lr = head_lr * backbone_scale * layer_decay**(max_id - i)，
        即最深层保持原 backbone lr，越靠近输入越小。layer_decay=1 与旧行为完全一致。
        ConvNeXt：max_id=7（见 _backbone_layer_id）；DINOv2：max_id=13（每个 block 一档）。
        """
        is_dino = self.arch.startswith("dinov2")
        if layer_decay != 1.0 and not (self.arch.startswith("convnext") or is_dino):
            raise ValueError("layer_decay 目前只支持 ConvNeXt / DINOv2")
        max_id = 13 if is_dino else 7
        if is_dino:
            from src.dinov2 import layer_id as dino_layer_id
        head_params, head_nodecay = [], []
        back = {}  # (layer_id, no_decay) -> params
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            is_head = name.startswith(("head.", "csra_head.", "view_emb.",
                                       "orient_emb.", "label_graph.", "pool.",
                                       "ml_decoder_head.", "fusion_"))
            no_decay = (p.ndim <= 1 or "LayerNorm" in name or "norm" in name.lower()
                        or name.endswith(("cls_token", "pos_embed")))
            if is_head:
                (head_nodecay if no_decay else head_params).append(p)
            else:
                if layer_decay == 1.0:
                    lid = max_id
                elif is_dino:
                    lid = dino_layer_id(name[len("body."):])
                else:
                    lid = self._backbone_layer_id(name)
                back.setdefault((lid, no_decay), []).append(p)
        groups = [
            {"params": head_params, "lr": head_lr, "weight_decay": weight_decay},
            {"params": head_nodecay, "lr": head_lr, "weight_decay": 0.0},
        ]
        for (lid, no_decay), params in sorted(back.items()):
            groups.append({
                "params": params,
                "lr": head_lr * backbone_scale * layer_decay ** (max_id - lid),
                "weight_decay": 0.0 if no_decay else weight_decay,
            })
        return [g for g in groups if g["params"]]


def build_model(arch: str, pretrained: bool = True, **kw) -> CXRModel:
    return CXRModel(arch=arch, pretrained=pretrained, **kw)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
