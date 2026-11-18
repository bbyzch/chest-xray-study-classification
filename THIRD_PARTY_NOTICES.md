# 权重来源与许可

仓库原创源码采用 [MIT License](LICENSE)。源模型与初始化的上游条款见下表及 [licenses/](licenses)。Release 提供参加项目训练后的四个学生模型。权重的使用仍须遵守其上游适用条款。

| 上游来源 | 用途 | 来源与条款 |
| --- | --- | --- |
| Meta ConvNeXt Base，ImageNet-22K 预训练后以 1K 微调 | 两个 Base 学生模型初始化 | [项目](https://github.com/facebookresearch/ConvNeXt)，[MIT](licenses/LICENSE-MIT-ConvNeXt.txt) |
| TorchVision ConvNeXt Tiny 和 Swin V2 Tiny，ImageNet-1K | Tiny、Swin 学生模型初始化 | [TorchVision v0.25.0](https://github.com/pytorch/vision/tree/v0.25.0)，[BSD-3-Clause](licenses/LICENSE-BSD-3-Clause-torchvision.txt) |
| TorchVision 的 ConvNeXt、Swin、EfficientNet（部分权重来自 timm / gen-efficientnet） | OOF 教师候选的上游模型 | [TorchVision](https://github.com/pytorch/vision)，[timm Apache 2.0](licenses/LICENSE-Apache-2.0-timm.txt)，[EfficientNet 上游 Apache 2.0](licenses/LICENSE-Apache-2.0-efficientnet-upstream.txt) |

TorchVision 在[预训练模型许可声明](https://github.com/pytorch/vision#pre-trained-model-license)中提示，预训练权重可能有来自上游数据的附加条款。许可证链接和代码许可仅用于帮助追溯。赛方材料及赛方数据的使用权限由[比赛规则](https://www.ncccu.org.cn/index/Paper/case1.html)和各自数据权利方决定。
