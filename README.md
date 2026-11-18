# 胸部 X 光检查级多标签分类

。本仓库提供训练和推理源码，训练后的四份权重可从 GitHub Release 下载。

| 模型 | 输入尺寸 | 集成权重 | 在线 CLAHE |
| --- | ---: | ---: | --- |
| ConvNeXt Base，ImageNet-22K 初始化 | 448 | 0.294 | 是 |
| Swin V2 Tiny | 384 | 0.253 | 是 |
| ConvNeXt Tiny | 384 | 0.231 | 否 |
| ConvNeXt Base，ImageNet-22K 初始化 | 384 | 0.222 | 是 |

四个学生以训练集折外预测作为软目标进行蒸馏，蒸馏权重为 0.6，温度为 2。推理先对成员概率加权，再取检查内多图均值；同患者其他检查可按固定系数参与平滑。类别次序在 [`src/labels.py`](src/labels.py) 中定义。

## 安装与预测

推荐 Python 3.11 或 3.12。CPU 示例使用官方的 PyTorch 2.10 / TorchVision 0.25 配对。GPU 用户可按 [PyTorch 安装页面](https://pytorch.org/get-started/locally/)选择对应的 CUDA 版本。

~~~bash
python -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu
~~~

~~~bash
python -m pip install -r requirements.txt
~~~

从 [v1.0.0 Release](https://github.com/bbyzch/chest-xray-study-classification/releases/tag/v1.0.0) 下载 `cxr-study-ensemble-v1.0.0.zip`，解压到仓库根目录，让 `weights/` 与 `predict.py` 同级。四份 SafeTensors 使用与原提交一致的张量。解压后的逐文件 SHA-256 在 `weights/SHA256SUMS.txt`。

用自己的本地图像和元数据运行。接受 JPG、JPEG 和 PNG。元数据需要以下七列。表格中的这一行只是字段示例。

~~~csv
Dicom_id,Study_id,Subject_id,Rows,Columns,ViewPosition,PatientOrientation
local_image_001,exam_a,person_a,512,512,PA,Erect
~~~

`Dicom_id` 为不带扩展名的文件名，需要唯一。`Rows`、`Columns` 为原图高度、宽度。视位或体位不明时可留空。每个检查须提供全部图像；默认患者平滑还需要提供元数据中该患者的全部检查。若仅有部分检查，使用 `--no-patient-smoothing`。

~~~bash
python predict.py --pictures path/to/local-images --metadata path/to/local-metadata.csv --out predictions.csv --device cpu
~~~

输出包含 `Study_id, Predict_class, Probability`，每个检查十行。训练阶段估计的归一化常量和集成配置放在 [`configs/`](configs)，因此预测不需要本地训练缓存。

Linux CUDA 环境可选择与原提交相同思路的 CUDA Graph 后端。它在 PyTorch 2.10.0 上使用 Inductor 编译三个 ConvNeXt，保留原生 Swin，并对每检查最多八张图的输入预热。该后端需要 Triton、`g++` 和 Python 开发头文件。原提交包的 T4 自测为 **87.82 ms/Study**。公开版在合成输入上验证了 CPU 预测，CUDA 延迟应在实际部署环境重新测量。

~~~bash
python predict.py --pictures path/to/local-images --metadata path/to/local-metadata.csv --out predictions-cuda.csv --device cuda --backend cuda-graph
~~~

## 本地训练

训练者需要自行准备有权限使用的图像和标注。可用 [`src/prepare_cache.py`](src/prepare_cache.py) 构建缓存，[`src/train.py`](src/train.py) 训练折外教师，[`tools/build_distill_targets.py`](tools/build_distill_targets.py) 汇总本地折外预测，[`tools/train_full_distill.py`](tools/train_full_distill.py) 训练全量学生。

~~~bash
python -m src.prepare_cache --data-root path/to/authorized-data --cache-dir cache --splits train --size 448
~~~

~~~bash
python tools/train_full_distill.py --arch convnext_base --image-size 448 --epochs 12 --stop-epoch 8 --batch-size 16 --pretrained-source imagenet22k --backbone-norm --distill-targets path/to/local-oof-teacher.npz --distill-alpha 0.6 --distill-temperature 2 --runs-dir runs_full_distill06
~~~

比赛图片、标签、逐样本元数据、折外软标签和原提交 CSV 由各自持有人保管。GitHub 发布内容为项目源码、模型配置和训练后权重。模型用于研究及比赛复现，尚未经临床验证。

原创代码许可见 [MIT License](LICENSE)，初始化权重来源与第三方条款见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
