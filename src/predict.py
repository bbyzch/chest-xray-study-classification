"""推理与提交生成：训练好的权重 → submission.csv。

赛制要求（官网「七、作品提交要求」）：
    每个 Study_id 下必须有 10 个类别 → 全网格 1414 × 10 = 14140 行
    文件命名 submission.csv，UTF-8

流程：
    1. 读 test 划分的缓存（图像级）
    2. 对每个 ckpt 跑推理（可选 hflip TTA），得到图像级概率
    3. 多模型求平均（集成）
    4. 图像级 → 检查级聚合（默认 mean，实测优于 max）
    5. 写 submission.csv 并做硬校验

用法：
    # 单折
    python -m src.predict --ckpts runs/convnext_tiny_s384_seed0/fold0

    # 5 折集成（通常比单折高 0.01 左右）
    python -m src.predict --ckpts "runs/convnext_tiny_s384_seed0/fold*" --tta

    # 多骨干集成
    python -m src.predict --ckpts "runs/*/fold*" --tta --out submission_ens.csv

注意：聚合用 mean 而不是 max。fold0 分析显示 mean 0.8061 > max 0.8008，
因为 max 会被单张误报图带偏。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.aggregate import (DEFAULT_STUDY_AGGREGATION, aggregate_study,
                           parse_patient_smooth, patient_smooth)  # noqa: E402
from src.dataset import CXRMemmapDataset  # noqa: E402
from src.labels import VIEW_VOCAB  # noqa: E402
from src.models import build_model  # noqa: E402
from src.prefetch import iter_studies  # noqa: E402
from src.submit import build_submission, write_submission  # noqa: E402
from src.train import StudyBatchSampler  # noqa: E402


def expand_ckpts(patterns) -> list[str]:
    """支持通配符，返回每个实验目录下的 ckpt.pt 路径。"""
    checkpoint_paths = []
    for p in patterns:
        matched_paths = sorted(glob.glob(p))
        if not matched_paths:
            raise FileNotFoundError(f"路径不存在: {p}")
        for m in matched_paths:
            ck = os.path.join(m, "ckpt.pt") if os.path.isdir(m) else m
            if not os.path.isfile(ck):
                raise FileNotFoundError(f"找不到 ckpt.pt: {ck}")
            checkpoint_paths.append(ck)
    return checkpoint_paths


def _cfg_from_ckpt(ckpt: dict) -> dict:
    """取出 ckpt 里保存的完整训练配置（dict）。"""
    import json as _json

    try:
        return _json.loads(ckpt.get("cfg") or "{}")
    except Exception:  # noqa: BLE001
        return {}


def _model_kwargs_from_cfg(ckpt: dict) -> dict:
    """从 ckpt 里保存的 cfg 还原**结构相关**的超参。

    必须做这一步：train.py 存 ckpt 时把完整 cfg 序列化进去了，但 predict.py
    以前只用 ckpt["arch"] 建模型，忽略了 pool / label_graph 这些会改变
    参数结构的开关。结果是加载 attention 池化或标签图的模型时报
    "Unexpected key(s) in state_dict: pool.score.*" 之类的错误。
    """
    cfg = _cfg_from_ckpt(ckpt)
    model_options = {}
    for k, default in (("pool", "gap"), ("csra_lambda", 0.5),
                       ("pretrained_source", "torchvision"),
                       ("backbone_norm", False), ("label_graph", False),
                       ("label_graph_alpha", 0.3), ("hidden", 256),
                       ("dropout", 0.3), ("view_dim", 8), ("orient_dim", 4),
                       ("study_fusion", False)):
        model_options[k] = cfg.get(k, default)
    model_options["study_fusion"] = cfg.get(
        "chex_fusion", cfg.get("study_fusion", False))
    return model_options


def _infer_kwargs_from_cfg(ckpt: dict) -> dict:
    """从 ckpt 里还原**推理预处理**相关的超参。

    ⚠ 这是曾经的一个真实 bug：predict.py 把 use_clahe 写死成 True，而
    `convnext_tiny_s384_seed0_nocl` 是**关掉 CLAHE 训练的**。于是训练看到的是
    「原始灰度」，推理却喂「CLAHE 增强图」——训练/推理分布错位，
    该模型在提交里的表现低于它 OOF 显示的水平，等于白占集成权重。

    现在按 ckpt 里的 cfg 还原，做到「怎么训的就怎么推」。
    """
    cfg = _cfg_from_ckpt(ckpt)
    return {"use_clahe": bool(cfg.get("use_clahe", True))}


def load_model(ckpt_path: str, device: torch.device):
    if os.fspath(ckpt_path).endswith(".safetensors"):
        from safetensors import safe_open

        with safe_open(ckpt_path, framework="pt", device="cpu") as archive:
            metadata = archive.metadata()
            ckpt = {
                "arch": metadata["arch"],
                "image_size": int(metadata["image_size"]),
                "cache_size": int(metadata["cache_size"]),
                "cfg": metadata["cfg"],
                "model": {key: archive.get_tensor(key) for key in archive.keys()},
            }
    else:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model = build_model(ckpt["arch"], pretrained=False, num_view=len(VIEW_VOCAB),
                        **_model_kwargs_from_cfg(ckpt))
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"{ckpt_path} 与当前模型结构不匹配：\n"
            f"  缺失参数 {list(missing)[:5]}\n"
            f"  多余参数 {list(unexpected)[:5]}\n"
            f"  （ckpt 里保存的 cfg 可能缺少结构字段，或代码版本不一致）")
    return model.to(device).eval(), ckpt


@torch.no_grad()
def infer(model, loader, device, tta_flip: bool = False) -> np.ndarray:
    """返回 (N,10) 图像级概率，行序与 index_test.csv 一致。"""
    probs = []
    fusion_mode = bool(getattr(model, "study_fusion", False))
    fusion_out = (np.zeros((len(loader.dataset), 10), dtype=np.float32)
                  if fusion_mode else None)
    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        view = batch["view_idx"].to(device, non_blocking=True)
        orient = batch["orient_idx"].to(device, non_blocking=True)
        size = batch["size_feat"].to(device, non_blocking=True)

        if fusion_mode:
            study_idx = batch["study_idx"].to(device, non_blocking=True)
            study_logits, _ = model.forward_study(
                image, view, orient, size, study_idx)
            if tta_flip:
                flip_study_logits, _ = model.forward_study(
                    image.flip(-1), view, orient, size, study_idx)
                study_logits = (study_logits + flip_study_logits) / 2
            starts = torch.ones_like(study_idx, dtype=torch.bool)
            if len(study_idx) > 1:
                starts[1:] = study_idx[1:] != study_idx[:-1]
            group = torch.cumsum(starts.to(torch.long), dim=0) - 1
            batch_probs = torch.sigmoid(study_logits.float())[group].cpu().numpy()
            fusion_out[batch["row_idx"].numpy()] = batch_probs
        else:
            logits = model(image, view, orient, size)
            if tta_flip:
                # 水平翻转 TTA。本题 10 个类别没有左右侧特异标签，所以安全
                logits = (logits + model(image.flip(-1), view, orient, size)) / 2
            probs.append(torch.sigmoid(logits.float()).cpu().numpy())
    return fusion_out if fusion_mode else np.concatenate(probs, axis=0)


def _parse_weights(spec: str | None, exp_dirs: list[str]) -> dict[str, float]:
    """解析 --weights，格式 "实验名:权重,实验名:权重"，例如
        --weights "convnext_tiny_s384_seed0:0.7,efficientnet_b0_s384_seed0:0.3"
    不传则各架构等权。权重会自动归一化到和为 1。
    """
    if not spec:
        return {e: 1.0 / len(exp_dirs) for e in exp_dirs}

    raw = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"权重格式应为 名称:权重，得到 {part!r}")
        name, val = part.rsplit(":", 1)
        raw[name.strip()] = float(val)

    missing = [e for e in exp_dirs if e not in raw]
    if missing:
        raise ValueError(
            f"这些实验没给权重: {missing}\n可用的实验名: {sorted(exp_dirs)}")
    total = sum(raw.values())
    if total <= 0:
        raise ValueError("权重之和必须大于 0")
    return {e: raw[e] / total for e in exp_dirs}


# ---------------------------------------------------------------------------
# 部署推理：按 Study 逐个推理。predict.py 的 --per-study 模式与 cost.py 共用以下函数，
# 保证 submission.csv 与计时脚本走同一条代码路径（规则第 7 条）。
# ---------------------------------------------------------------------------

def load_deploy(path: str) -> dict:
    """读部署清单 deploy.json；其中的相对路径以 deploy.json 所在目录为基准。

    字段：ckpts（ckpt 路径列表，可含通配符）、weights（实验名→权重）、
    precision（fp32/fp16）、meta_csv（测试集元数据，含 ViewPosition 等）、
    patient_smooth（可选，10 个逐类 α 的字符串）。
    """
    with open(path, encoding="utf-8") as f:
        cfg = json.load(f)
    base = os.path.dirname(os.path.abspath(path))
    resolve = lambda p: p if os.path.isabs(p) else os.path.join(base, p)  # noqa: E731
    cfg["ckpts"] = [resolve(p) for p in cfg["ckpts"]]
    for key in ("meta_csv", "norm_json"):
        if cfg.get(key):
            cfg[key] = resolve(cfg[key])
    cfg.setdefault("precision", "fp32")
    return cfg


class Ensemble:
    """一次性加载全部成员；权重按实验名（ckpt 上两级目录名）归一化。"""

    def __init__(self, ckpt_patterns, weights: dict | str | None, device,
                 precision: str = "fp32"):
        if precision not in ("fp32", "fp16"):
            raise ValueError(f"precision 只能是 fp32/fp16，得到 {precision!r}")
        self.device = torch.device(device)
        self.precision = precision
        paths = expand_ckpts(ckpt_patterns)
        names = [os.path.basename(os.path.dirname(os.path.dirname(p))) for p in paths]
        spec = (",".join(f"{k}:{v}" for k, v in weights.items())
                if isinstance(weights, dict) else weights)
        w = _parse_weights(spec, sorted(set(names)))
        counts = {n: names.count(n) for n in set(names)}
        self.members = []
        for path, name in zip(paths, names):
            model, ck = load_model(path, self.device)
            self.members.append({
                "name": name, "model": model,
                "weight": w[name] / counts[name],      # 同一实验多个 ckpt 等分权重
                "image_size": int(ck.get("image_size", 384)),
                "use_clahe": _infer_kwargs_from_cfg(ck)["use_clahe"],
            })

    def describe(self) -> str:
        return ", ".join(f"{m['name']}={m['weight']:.3f}" for m in self.members)

    def autocast(self):
        return torch.autocast(device_type=self.device.type, dtype=torch.float16,
                              enabled=(self.precision == "fp16" and self.device.type == "cuda"))

    @torch.no_grad()
    def warmup(self, max_images: int = 8) -> None:
        """用全零输入把每个成员在 batch=1..max_images 上各前向一次。

        GPU 第一次遇到某个输入形状时要选卷积算法、分配显存，实测单个 Study 可多花数秒；
        这属于规则第 7 条不计时的"首次初始化"。只影响速度，不影响输出：
        cudnn.benchmark 关闭时，算法由形状决定，与之前是否跑过无关。
        max_images 取固定常数（训练集单个 Study 最多 7 张），不依赖测试集。
        """
        for m in self.members:
            s = m["image_size"]
            for b in range(1, max_images + 1):
                x = torch.zeros(b, 1, s, s, device=self.device)
                z = torch.zeros(b, dtype=torch.long, device=self.device)
                f = torch.zeros(b, 2, device=self.device)
                with self.autocast():
                    m["model"](x, z, z, f)
        if self.device.type == "cuda":
            torch.cuda.synchronize()


def load_letterbox(path: str, size: int = 448) -> np.ndarray:
    """原始图片 → 与缓存逐字节一致的 uint8 letterbox（prepare_cache 的同一函数）。"""
    from src.prepare_cache import load_gray, preprocess
    return preprocess(load_gray(path), size)


def study_rows(meta_rows: pd.DataFrame) -> pd.DataFrame:
    """把测试集元数据行整理成 Dataset 需要的列（view_idx/orient_idx 缺失时现场计算）。"""
    from src.labels import orient_index, view_index
    rows = meta_rows.reset_index(drop=True).copy()
    if "view_idx" not in rows:
        rows["view_idx"] = [view_index(v) for v in rows["ViewPosition"]]
    if "orient_idx" not in rows:
        rows["orient_idx"] = [orient_index(v) for v in rows["PatientOrientation"]]
    rows["row_idx"] = np.arange(len(rows))
    return rows


EXTS = (".jpg", ".jpeg", ".png")


def collect_studies(root: str, meta: pd.DataFrame) -> list[tuple[str, list[str], pd.DataFrame]]:
    """扫描图片目录，按文件名（Dicom_id）对元数据，再按 Study_id 分组。

    评测机只给原始图片，这条路径与 cost.py 的计时口径完全一致。
    """
    by_dicom = meta.set_index("Dicom_id")
    files = []
    for d, _, names in os.walk(root):
        files += [os.path.join(d, n) for n in names if n.lower().endswith(EXTS)]
    if not files:
        raise FileNotFoundError(f"{root} 下没有图片")
    dicoms = [os.path.splitext(os.path.basename(f))[0] for f in files]
    missing = [x for x in dicoms if x not in by_dicom.index]
    if missing:
        raise KeyError(f"{len(missing)} 张图不在元数据里，例如 {missing[:3]}")
    df = by_dicom.loc[dicoms].reset_index()
    df["path"] = files
    studies = []
    for sid, g in df.groupby("Study_id", sort=True):
        g = g.sort_values("Dicom_id")
        studies.append((sid, g["path"].tolist(), study_rows(g)))
    return studies


@torch.no_grad()
def predict_study(ens: Ensemble, images: np.ndarray, rows: pd.DataFrame,
                  norm: dict, cache_size: int = 448,
                  timing: dict | None = None) -> np.ndarray:
    """一个 Study 的全部图片（N,448,448 uint8）→ 10 维检查级概率。

    每个成员把该 Study 的 N 张图作为一个 batch 前向；图像级概率按成员权重加权，
    再在 Study 内取 mean（与训练/评估的聚合口径一致）。所有图片都参与，
    分辨率和成员固定，不做任何按 Study 内容的自适应（官方答疑）。

    速度上的两点（都不改变数值）：
      * 预处理配置（分辨率, CLAHE）相同的成员共用一份输入，只算一次；
      * 先把全部成员的前向依次排进 GPU 队列，最后统一取回，CPU 不在成员之间等 GPU。
    timing 给出时累加 "preprocess" / "gpu" 两段耗时（秒）。
    """
    if len(images) != len(rows):
        raise ValueError(f"图片数 {len(images)} 与元数据行数 {len(rows)} 不一致")
    t0 = time.perf_counter()
    inputs = {}
    for m in ens.members:
        key = (m["image_size"], m["use_clahe"])
        if key in inputs:
            continue
        ds = CXRMemmapDataset(
            arr_path="<in-memory>", rows=rows, image_size=key[0], train=False,
            mean=norm["mean"], std=norm["std"], labels=None, use_clahe=key[1],
            size_mean=norm["size_mean"], size_std=norm["size_std"], cache_size=cache_size)
        ds._arr = images                      # 绕过 memmap，直接用内存中的图
        items = [ds[i] for i in range(len(ds))]
        inputs[key] = {k: torch.stack([it[k] for it in items]).to(ens.device)
                       for k in ("image", "view_idx", "orient_idx", "size_feat")}
    t1 = time.perf_counter()
    logits = []
    with ens.autocast():
        for m in ens.members:
            b = inputs[(m["image_size"], m["use_clahe"])]
            logits.append(m["model"](b["image"], b["view_idx"],
                                     b["orient_idx"], b["size_feat"]))
    probs = [torch.sigmoid(z.float()) for z in logits]
    probs = [p.cpu().numpy().astype(np.float64) for p in probs]   # 第一次 .cpu() 处同步 GPU
    t2 = time.perf_counter()
    total = None
    for m, p in zip(ens.members, probs):
        p = p * m["weight"]
        total = p if total is None else total + p
    if timing is not None:
        timing["preprocess"] = timing.get("preprocess", 0.0) + (t1 - t0)
        timing["gpu"] = timing.get("gpu", 0.0) + (t2 - t1)
    return total.mean(axis=0)


def run_per_study(ens: Ensemble, meta: pd.DataFrame, arr, norm: dict,
                  cache_size: int = 448) -> tuple[np.ndarray, np.ndarray]:
    """在缓存上按 Study 逐个推理，返回 (按 Study_id 排序的概率, Study_id)。"""
    study_ids = np.array(sorted(meta["Study_id"].unique()))
    out = np.zeros((len(study_ids), 10))
    for i, sid in enumerate(study_ids):
        rows = meta[meta["Study_id"] == sid]
        images = np.stack([np.asarray(arr[r]) for r in rows["row_idx"]])
        out[i] = predict_study(ens, images, study_rows(rows), norm, cache_size)
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(study_ids)} studies", flush=True)
    return out, study_ids


def main_per_study(args, meta: pd.DataFrame, norm: dict, arr_path: str) -> int:
    """--deploy 模式：按部署清单逐 Study 推理并写 submission.csv。"""
    dep = load_deploy(args.deploy)
    ens = Ensemble(dep["ckpts"], dep["weights"], args.device, dep["precision"])
    print(f"部署清单 {args.deploy}  精度 {dep['precision']}\n成员: {ens.describe()}")
    arr = np.load(arr_path, mmap_mode="r")
    probs, study_ids = run_per_study(ens, meta, arr, norm, args.cache_size)
    if args.save_probs:
        np.savez_compressed(args.save_probs, probs=probs.astype(np.float32),
                            study_id=study_ids.astype(str))
    if dep.get("patient_smooth"):
        raw_path = os.path.splitext(args.out)[0] + "_nosmooth.csv"
        raw_sub = build_submission(study_ids, probs, full_grid=True)
        write_submission(raw_sub, raw_path, expected_study_ids=study_ids, full_grid=True)
        print(f"平滑前结果另存 {raw_path}（cost.py 用它做逐 Study 一致性比对）")
        st2sub = meta.groupby("Study_id")["Subject_id"].first()
        probs = patient_smooth(probs, st2sub.loc[study_ids].to_numpy(),
                               parse_patient_smooth(dep["patient_smooth"]))
        print(f"患者平滑: {dep['patient_smooth']}")
    sub = build_submission(study_ids, probs, full_grid=True)
    write_submission(sub, args.out, expected_study_ids=study_ids, full_grid=True)
    print(f"已写出 {args.out}：{len(study_ids)} 个检查，格式校验通过 ✅")
    return 0


def main_from_pictures(args, meta: pd.DataFrame, norm: dict, pictures: str) -> int:
    """--pictures 模式：原始图片目录 → study 级概率 → submission.csv。

    评测机只给图片和 test.csv，不给缓存；这条路径与 cost.py 共用 predict_study
    和预解码，所以 cost.py 量到的耗时就是提交版本的真实耗时。
    """
    dep = load_deploy(args.deploy)
    ens = Ensemble(dep["ckpts"], dep["weights"], args.device, dep["precision"])
    print(f"部署清单 {args.deploy}  精度 {dep['precision']}\n成员: {ens.describe()}")
    studies = collect_studies(pictures, meta)
    n_img = sum(len(p) for _, p, _ in studies)
    print(f"原始图片: {len(studies)} 个 Study / {n_img} 张  "
          f"预解码 {'关' if args.no_prefetch else '开'}")
    # 首次遇到某个 batch 形状时 GPU 要选算法；预热掉它，下面打印的 ms/Study 才是部署耗时
    t_warm = time.perf_counter()
    ens.warmup(8)
    print(f"形状预热 batch=1..8 完成（{time.perf_counter() - t_warm:.1f}s，不计时）")

    pool = ThreadPoolExecutor(max_workers=args.num_workers)
    decode = lambda paths: np.stack(list(pool.map(load_letterbox, paths)))  # noqa: E731
    stream = iter_studies(studies, decode, use_prefetch=not args.no_prefetch,
                          timing=None)
    sids, probs = [], []
    t0 = time.perf_counter()
    for sid, _, rows, images in stream:
        probs.append(predict_study(ens, images, rows, norm, args.cache_size))
        sids.append(sid)
        if len(sids) % 200 == 0:
            print(f"  {len(sids)}/{len(studies)} studies", flush=True)
    pool.shutdown()
    probs = np.vstack(probs)
    elapsed = time.perf_counter() - t0
    print(f"推理完成: {len(sids)} 个 Study，{elapsed / len(sids) * 1000:.1f} ms/Study")

    study_ids = np.array(sids)
    if args.save_probs:
        np.savez_compressed(args.save_probs, probs=probs.astype(np.float32),
                            study_id=study_ids.astype(str))
    if dep.get("patient_smooth"):
        st2sub = meta.groupby("Study_id")["Subject_id"].first()
        probs = patient_smooth(probs, st2sub.loc[study_ids].to_numpy(),
                               parse_patient_smooth(dep["patient_smooth"]))
        print(f"患者平滑: {dep['patient_smooth']}")
    sub = build_submission(study_ids, probs, full_grid=True)
    write_submission(sub, args.out, expected_study_ids=study_ids, full_grid=True)
    print(f"已写出 {args.out}：{len(study_ids)} 个检查，格式校验通过 ✅")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", default=None,
                   help="ckpt.pt 路径或实验目录，支持通配符，可给多个")
    p.add_argument("--cache-dir", default="cache")
    p.add_argument("--cache-size", type=int, default=448)
    p.add_argument("--split", default="test")
    p.add_argument("--agg", default=DEFAULT_STUDY_AGGREGATION,
                   choices=["mean", "max", "logsumexp"],
                   help="图像→检查聚合方式，默认 mean（实测优于 max）")
    p.add_argument("--tta", action="store_true", help="启用 hflip TTA")
    p.add_argument("--weights", default=None,
                   help='各架构权重，如 "convnext_tiny_s384_seed0:0.7,'
                        'efficientnet_b0_s384_seed0:0.3"；不传则等权')
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="submission.csv")
    p.add_argument("--save-probs", default=None, help="把图像级概率存成 npz")
    p.add_argument("--patient-smooth", default=None,
                   help="患者平滑系数：1 个值或 10 个逗号分隔值（由 "
                        "tools/fit_patient_smooth.py 在训练 OOF 上选定）；默认关闭")
    p.add_argument("--deploy", default=None,
                   help="部署清单 deploy.json：按 Study 逐个推理，与 cost.py 同一代码路径。"
                        "给出时忽略 --ckpts/--weights/--tta/--patient-smooth")
    p.add_argument("--pictures", default=None,
                   help="原始图片目录：不读缓存，逐 Study 读图解码（评测机用这条路）")
    p.add_argument("--data-root", default=None,
                   help="数据集根目录（含 test.csv 与 test/）；给出时元数据和图片目录都由它推导")
    p.add_argument("--meta-csv", default=None,
                   help="--pictures 模式下的元数据 CSV，默认取部署清单的 meta_csv")
    p.add_argument("--no-prefetch", action="store_true",
                   help="关闭 Study 级预解码（只用于与串行做对照）")
    args = p.parse_args(argv)
    if not args.deploy and not args.ckpts:
        p.error("需要 --ckpts 或 --deploy")
    if (args.pictures or args.data_root) and not args.deploy:
        p.error("--pictures/--data-root 需要配合 --deploy（成员、权重、平滑系数都在清单里）")
    smooth_alphas = parse_patient_smooth(args.patient_smooth)

    # 评测机没有缓存：元数据从 test.csv 推导，图片从 <data-root>/test 读
    if args.deploy and (args.pictures or args.data_root):
        dep = load_deploy(args.deploy)
        if args.data_root:
            from src.labels import load_split_df
            meta = load_split_df("test", args.data_root, need_labels=False)
            pictures = args.pictures or os.path.join(args.data_root, "test")
        else:
            meta = pd.read_csv(args.meta_csv or dep["meta_csv"])
            pictures = args.pictures
        with open(dep["norm_json"], encoding="utf-8") as f:
            norm = json.load(f)
        return main_from_pictures(args, meta, norm, pictures)

    # 数据
    index_csv = os.path.join(args.cache_dir, f"index_{args.split}.csv")
    arr_path = os.path.join(args.cache_dir, f"{args.split}_{args.cache_size}.npy")
    for f in (index_csv, arr_path):
        if not os.path.isfile(f):
            raise FileNotFoundError(f"缺少缓存文件: {f}（先跑 prepare_cache）")

    meta = pd.read_csv(index_csv)
    with open(os.path.join(args.cache_dir, "norm.json"), encoding="utf-8") as f:
        norm = json.load(f)

    if args.deploy:
        return main_per_study(args, meta, norm, arr_path)

    device = torch.device(args.device)
    ckpts = expand_ckpts(args.ckpts)
    print(f"设备: {device}   模型数: {len(ckpts)}   TTA: {args.tta}   聚合: {args.agg}")
    for c in ckpts:
        print(f"  - {c}")

    # 逐模型推理
    # 按"实验目录"（架构+seed）分组：同一实验的多个折先平均，得到一个架构级预测，
    # 再按权重合并各架构。这样权重是"每个架构一个"，与 analyze_oof 搜出来的一致。
    groups: dict[str, list[str]] = {}
    for ck in ckpts:
        exp_dir = os.path.basename(os.path.dirname(os.path.dirname(ck)))
        groups.setdefault(exp_dir, []).append(ck)

    weights = _parse_weights(args.weights, list(groups))
    print(f"\n集成权重: " + ", ".join(f"{k}={weights[k]:.3f}" for k in groups))

    sum_probs = None
    done = 0
    for gi, (exp_dir, ck_list) in enumerate(sorted(groups.items()), 1):
        arch_sum = None
        for ck in ck_list:
            done += 1
            model, ckpt = load_model(ck, device)
            image_size = int(ckpt.get("image_size", 384))
            # ⚠ 必须按训练时的预处理推理：nocl 模型是关 CLAHE 训的，
            # 如果这里写死 True，训练/推理分布错位，该模型输出会劣化。
            inf = _infer_kwargs_from_cfg(ckpt)
            print(f"\n[{done}/{len(ckpts)}] {exp_dir}/{os.path.basename(os.path.dirname(ck))}"
                  f"  arch={ckpt['arch']} size={image_size} clahe={inf['use_clahe']}")

            ds = CXRMemmapDataset(
                arr_path=arr_path, rows=meta, image_size=image_size, train=False,
                mean=norm["mean"], std=norm["std"], labels=None,
                use_clahe=inf["use_clahe"],
                size_mean=norm["size_mean"], size_std=norm["size_std"],
                cache_size=args.cache_size,
            )
            if bool(getattr(model, "study_fusion", False)):
                sampler = StudyBatchSampler(ds.study_idx, args.batch_size,
                                            seed=0, shuffle=False)
                loader = DataLoader(ds, batch_sampler=sampler,
                                    num_workers=args.num_workers,
                                    pin_memory=(device.type == "cuda"))
            else:
                loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers,
                                    pin_memory=(device.type == "cuda"))
            p = infer(model, loader, device, tta_flip=args.tta)
            arch_sum = p if arch_sum is None else arch_sum + p
            del model, loader, ds
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # 该架构内部：各折等权平均
        arch_probs = arch_sum / len(ck_list)
        w = weights[exp_dir]
        sum_probs = w * arch_probs if sum_probs is None else sum_probs + w * arch_probs
        print(f"    {exp_dir}: {len(ck_list)} 折平均，权重 {w:.3f}")

    probs = sum_probs

    # 聚合到检查级
    study_probs, study_ids = aggregate_study(probs, meta["Study_id"].to_numpy(),
                                             mode=args.agg)
    print(f"\n聚合（{args.agg}）: {len(probs)} 张图 → {len(study_ids)} 个检查")

    if smooth_alphas is not None:
        study_to_subject = meta.groupby("Study_id")["Subject_id"].first()
        subjects = study_to_subject.loc[study_ids].to_numpy()
        study_probs = patient_smooth(study_probs, subjects, smooth_alphas)
        print(f"患者平滑: α={np.round(smooth_alphas, 3).tolist()}  "
              f"患者数 {len(np.unique(subjects))}")

    if args.save_probs:
        np.savez_compressed(args.save_probs, probs=probs.astype(np.float32),
                            study_id=np.asarray(meta["Study_id"].to_numpy(), dtype=str))
        print(f"图像级概率已保存: {args.save_probs}")

    # 写提交
    sub = build_submission(study_ids, study_probs, full_grid=True)
    write_submission(sub, args.out, expected_study_ids=study_ids, full_grid=True)

    print(f"\n{'=' * 60}")
    print(f"已写出 {args.out}")
    print(f"  行数 {len(sub)}   检查 {sub['Study_id'].nunique()}   类别 {sub['Predict_class'].nunique()}")
    print(f"  概率范围 [{sub['Probability'].min():.4f}, {sub['Probability'].max():.4f}]")
    print(f"  格式校验通过 ✅（每个 Study_id 恰好 10 行）")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
