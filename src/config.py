"""单一配置源。

训练脚本用 dataclass 承载全部超参：
  - 好处一：新增参数有默认值，老的实验命令不会因为加字段而失效；
  - 好处二：实验目录里会落一份 JSON（TrainConfig.to_json），便于事后复现。

路径全部是**相对路径**：缓存建好后要上传到 4090 机器，
那边只有 cache/ 目录、没有 55GB 原始数据集，写死绝对路径会直接失效。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json


@dataclass
class TrainConfig:
    # 数据
    cache_dir: str = "cache"          # prepare_cache.py 的产物目录
    runs_dir: str = "runs"            # 每次实验的输出根目录
    cache_size: int = 448             # 缓存边长（prepare_cache --size）
    image_size: int = 384             # 训练/推理分辨率，可与缓存不同
    use_clahe: bool = True            # 对比度受限直方图均衡，CXR 上通常有正收益

    # 模型
    arch: str = "convnext_tiny"       # 必须是 models.ARCHS 里的键
    pretrained: bool = True           # ImageNet 预训练；无网络时设 False 仅用于冒烟
    pretrained_source: str = "torchvision"  # torchvision / imagenet22k
    dropout: float = 0.3
    view_dim: int = 8                 # ViewPosition embedding 维度
    orient_dim: int = 4               # PatientOrientation embedding 维度
    hidden: int = 256                 # 分类头隐层宽度
    pool: str = "gap"                 # gap / gem / attention / csra
    csra_lambda: float = 0.5           # CSRA max-pool residual weight
    backbone_norm: bool = False       # 保留预训练末端 norm；默认兼容旧权重
    label_graph: bool = False         # 启用标签图细化（建模标签共现）
    label_graph_alpha: float = 0.3    # 耦合强度
    label_graph_l1: float = 1e-3      # 对耦合矩阵的 L1 系数（促稀疏）

    # 数据增强强度：normal 是原方案；strong 用于缓解"ep7 就过拟合"
    aug: str = "normal"

    # 损失：bce / asl / auc（BCE 预训练后 LibAUC 微调）
    loss: str = "bce"
    asl_gamma_neg: float = 4.0
    asl_gamma_pos: float = 0.0
    asl_clip: float = 0.05
    # LibAUC 两阶段训练：前 auc_start_epoch 轮用 BCE，之后用 MultiLabelAUCMLoss+PESG。
    auc_start_epoch: int = 6
    auc_lr: float = 1e-4
    auc_margin: float = 1.0
    auc_version: str = "v1"
    auc_weight_decay: float = 1e-5
    auc_epoch_decay: float = 2e-3
    # 标签平滑：针对"标签由 NLP 从报告抽取、类 4/8 本身模糊"这个根本噪声来源。
    # 0 = 原行为（硬标签）。他人方案用 0.1。
    label_smoothing: float = 0.0
    # 冲突样本降权：cvs 来自 tools/find_conflicts.py --export（检查级）。
    # 空 = 不降权（原行为）。配合 conflict_weight 使用，常用 0.3。
    conflicts_csv: str | None = None
    conflict_weight: float = 0.3
    # 按检查归一化：每张图权重 = 1/该检查图片数，使每个检查对总损失贡献恒等。
    # 对齐「评估是检查级、训练是图像级」这个口径差。False = 原行为。
    study_balanced: bool = False
    # 按检查完整分组训练：同一 Study 的多张图先做 mean pooling，再计算损失。
    study_mil: bool = False
    # CheXFusion-lite：Study 内特征级 Transformer 融合，要求 pool=gap。
    chex_fusion: bool = False
    # 监督蒸馏：训练图像的 OOF 教师软目标路径；空值表示普通监督。
    distill_targets: str | None = None
    distill_alpha: float = 0.3
    distill_temperature: float = 2.0

    # 优化
    batch_size: int = 64
    epochs: int = 12                  # 2.9 万图，12 epoch 通常足够收敛
    lr: float = 3e-4                  # head 的学习率
    backbone_scale: float = 0.3       # backbone 用 lr*0.3，预训练特征只做微调
    # 逐层学习率衰减（LLRD，仅 ConvNeXt）。1.0 = 旧行为（主干统一 lr）。
    layer_decay: float = 1.0
    # stochastic depth 覆盖值。None = torchvision 默认（convnext_base 为 0.5，
    # 是从头训练的配方，对 3 万张图的微调可能过强）。
    drop_path: float | None = None
    weight_decay: float = 0.05
    warmup_epochs: int = 1            # 线性 warmup，避免初期把预训练特征冲坏
    ema_decay: float = 0.999          # 推理一律用 EMA 权重
    grad_clip: float = 5.0

    # 损失
    # 实测 LATERAL 有 62% 是「无异常」，标签噪声最大（报告结论被复制到侧位图），
    # 所以给侧位样本降权。1.0 等于不降权，可在 OOF 上消融。
    lateral_weight: float = 0.5
    # 默认不启用正样本加权：指标是 AUC（排序），过度加权会破坏稀有类的排序质量
    use_pos_weight: bool = False
    pos_weight_power: float = 0.5     # pos_weight = (neg/pos)^power，再截断
    pos_weight_cap: float = 8.0

    # 运行
    num_splits: int = 5               # 交叉验证折数
    num_workers: int = 6              # 本机内存有限时调小；受限环境可设 0
    seed: int = 0                     # 数据划分种子：决定 5 折怎么分（集成必须一致）
    init_seed: int | None = None      # 模型初始化种子；None 表示跟随 seed。
                                      # 做"同架构不同初始化"的多 seed 集成时，
                                      # 必须固定 seed、只改 init_seed，否则分折会变、
                                      # OOF 覆盖的检查不同，集成时会被判为不一致而跳过。
    amp: bool = True                  # 混合精度；T4 走 fp16，4090 走 bf16
    device: str = "cuda"
    limit_images: int | None = None   # 调试用：只取前 N 张
    save_oof: bool = True             # 是否落 oof.npz（集成阶段依赖它）
    extra: dict = field(default_factory=dict)   # 临时塞实验备注

    def to_json(self) -> str:
        """序列化到实验目录，便于复现与对比。ensure_ascii=False 让中文可读。"""
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @property
    def exp_name(self) -> str:
        """实验目录名。

        **必须把所有影响结果的变量都编进名字**，否则跑消融时会覆盖掉基线，
        比如 `--loss asl` 和普通 BCE 都叫 convnext_tiny_s384_seed0，
        后跑的会把先跑的结果删掉，而且不报错。
        只有非默认值的开关才加后缀，保持基线目录名不变（兼容已有结果）。
        """
        name = f"{self.arch}_s{self.image_size}_seed{self.seed}"
        tags = []
        if self.pretrained_source == "imagenet22k":
            tags.append("pre22k")
        elif self.pretrained_source == "clip_laion":
            tags.append("clip")
        elif self.pretrained_source == "ssl_cxr":
            tags.append("ssl")
        elif self.pretrained_source != "torchvision":
            tags.append(self.pretrained_source)
        # 多初始化集成：seed 固定，用 iseed 后缀区分，保证分折一致、OOF 可对齐
        if self.init_seed is not None and self.init_seed != self.seed:
            tags.append(f"iseed{self.init_seed}")
        if self.loss != "bce":
            tags.append(self.loss)
        if self.label_smoothing > 0:
            tags.append(f"ls{self.label_smoothing:g}")
        # 冲突降权必须进名字，否则会覆盖基线的实验目录
        if self.conflicts_csv:
            tags.append(f"cw{self.conflict_weight:g}")
        if self.study_balanced:
            tags.append("sb")
        if self.study_mil:
            tags.append("mil")
        if self.chex_fusion:
            tags.append("chexfusion")
        if self.distill_targets:
            tags.append(f"distill{self.distill_alpha:g}t{self.distill_temperature:g}")
        if self.pool != "gap":
            tags.append(self.pool if self.pool != "csra"
                        else f"csra{self.csra_lambda:g}")
        if self.backbone_norm:
            tags.append("bnorm")
        if self.label_graph:
            tags.append("lg")
        if self.aug != "normal":
            tags.append(self.aug)
        if not self.use_clahe:
            tags.append("nocl")
        if self.lateral_weight != 0.5:
            tags.append(f"lat{self.lateral_weight:g}")
        # pos_weight 必须进名字，否则会覆盖基线目录（原来漏了）
        if self.use_pos_weight:
            tags.append(f"pw{self.pos_weight_power:g}")
        if self.drop_path is not None:
            tags.append(f"dp{self.drop_path:g}")
        if self.layer_decay != 1.0:
            tags.append(f"ld{self.layer_decay:g}")
        if self.lr != 3e-4:
            tags.append(f"lr{self.lr:g}")
        if self.backbone_scale != 0.3:
            tags.append(f"bs{self.backbone_scale:g}")
        if self.epochs != 12:
            tags.append(f"e{self.epochs}")
        return name + ("_" + "-".join(tags) if tags else "")
