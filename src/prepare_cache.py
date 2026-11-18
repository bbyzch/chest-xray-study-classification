"""缓存预处理：原始 JPG → 448×448 灰度 letterbox → memmap。

为什么要做这一步（设计文档 §4.2）：
原图普遍 2000~3000 px，每个 epoch 现解码 3 万张 JPEG 是分钟级的 CPU 开销，
会把训练完全卡在 IO 上。预先把全部图片缩放成 448×448 的 uint8 memmap，
训练时只需一次 O(1) 的切片，吞吐由 GPU 决定。

产出（全部落在 cache/）：
    {split}_{size}.npy    (N, size, size) uint8 memmap，约 6.8GB（三划分合计）
    index_{split}.csv     行号 ↔ Dicom_id/Study_id/Subject_id/view/orient/size
    {split}_labels.npy    (N, 10) uint8，测试集不产出
    norm.json             mean/std/size_mean/size_std

云端只需上传 cache/ 目录（约 6.8GB），不必上传 55GB 原始数据。

用法：
    python -m src.prepare_cache --data-root path/to/local-data --workers 6 --size 448
    python -m src.prepare_cache --data-root path/to/local-data --limit 400 --workers 1   # 调试
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import get_context
from typing import Sequence

import numpy as np

# 允许 `python -m src.prepare_cache` 与 `python src/prepare_cache.py` 两种跑法
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.labels import load_split_df, labels_matrix  # noqa: E402

# index CSV 的固定列序。row_idx 必须与 .npy 的第一维严格对应，
# 训练时靠它把概率映射回 Study_id，行序错一位就会全盘错位。
INDEX_COLUMNS = [
    "row_idx", "Dicom_id", "Subject_id", "Study_id", "split", "relpath",
    "ViewPosition", "PatientOrientation", "view_idx", "orient_idx", "Rows", "Columns",
]


def load_gray(path: str) -> np.ndarray:
    """读成单通道 uint8。兼容灰度 JPEG 与意外出现的 RGB/16bit 图。

    **不能用 cv2.imread**：Windows 上 OpenCV 走 ANSI 文件 API，路径只要含非 ASCII
    字符就静默返回 None，最终得到一堆全黑图而且不报错。
    改用 Python 读字节 + cv2.imdecode，在中文路径、Linux、容器里行为一致。
    """
    import cv2

    # np.fromfile 走 Python 的宽字符路径 API，能正确处理中文路径
    try:
        encoded_image = np.fromfile(path, dtype=np.uint8)
    except OSError as exc:
        raise IOError(f"无法读取文件: {path} ({exc})") from exc
    if encoded_image.size == 0:
        raise IOError(f"空文件: {path}")

    # 从内存字节解码，绕开 OpenCV 的路径处理
    img = cv2.imdecode(encoded_image, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"无法解码图片: {path}")
    # 极少数 16bit/异常位深的图，统一线性拉伸到 0~255
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img


def letterbox(img: np.ndarray, size: int) -> np.ndarray:
    """等比缩放到 (size, size)，不足处以 0 居中填充，保持解剖比例不失真。

    为什么不直接 cv2.resize 到正方形：原图宽高比各异（2381×2654、2021×2021…），
    强行拉伸会让心脏、膈肌等解剖结构的形状发生畸变，模型学到的形态特征失真。
    """
    import cv2

    if img.ndim != 2:
        raise ValueError(f"期望二维灰度图, 得到 {img.shape}")
    h, w = img.shape
    if h == 0 or w == 0:
        raise ValueError("空图")

    # 取长边作为缩放基准，保证短边不会超出画布
    scale = size / float(max(h, w))
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))

    # 缩小用 INTER_AREA（抗锯齿、更接近真实降采样），放大用 INTER_CUBIC（更平滑）
    resize_interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    scaled_image = cv2.resize(img, (nw, nh), interpolation=resize_interpolation)

    # 居中贴到全黑画布上
    out = np.zeros((size, size), dtype=np.uint8)
    top = (size - nh) // 2
    left = (size - nw) // 2
    out[top:top + nh, left:left + nw] = scaled_image
    return out


def crop_black_border(img: np.ndarray, thresh: int = 30, min_ratio: float = 0.01,
                      margin: float = 0.02, min_keep: float = 0.20) -> np.ndarray:
    """裁掉探测器黑边，回收有效分辨率。

    参数（都经过保守设定，宁可少裁不要误裁）：
      thresh     像素值 < thresh 视为背景黑
      min_ratio  某一行/列中"非背景像素"占比 > min_ratio 才认为该行/列有效
      margin     裁完后四周保留 margin 比例的余量，防止切掉肺尖/肋膈角
      min_keep   若裁完任一边小于原来的 min_keep，判定异常直接放弃裁剪

    为什么保守：肺尖（气胸）和肋膈角（积液）正是我们最关心的区域，
    裁过头会把这些关键部位切掉。实测原图 letterbox 后有效面积已有 85%，
    所以这项的潜在收益只有 5~10%，不值得冒险。
    """
    h, w = img.shape
    foreground_mask = img >= thresh
    rows = foreground_mask.mean(axis=1) > min_ratio
    cols = foreground_mask.mean(axis=0) > min_ratio
    if not rows.any() or not cols.any():
        return img

    top = int(np.argmax(rows))
    bottom = h - int(np.argmax(rows[::-1]))
    left = int(np.argmax(cols))
    right = w - int(np.argmax(cols[::-1]))

    # 加回 margin
    dy, dx = int((bottom - top) * margin), int((right - left) * margin)
    top, bottom = max(0, top - dy), min(h, bottom + dy)
    left, right = max(0, left - dx), min(w, right + dx)

    # 防误裁：任一边剩余不足 min_keep 就放弃
    if (bottom - top) < h * min_keep or (right - left) < w * min_keep:
        return img
    return img[top:bottom, left:right]


def normalize_contrast(img: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.0) -> np.ndarray:
    """百分位截断线性拉伸：先稳住动态范围，CLAHE 才不会放大噪声。

    为什么需要：不同设备/批次的图整体亮度差异明显。不先截断就上 CLAHE，
    极端亮的区域会被当成信号反复增强。
    """
    lo, hi = np.percentile(img, [p_lo, p_hi])
    if hi - lo < 1e-6:
        return img
    out = (img.astype(np.float32) - lo) / (hi - lo) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_clahe(img: np.ndarray, clip: float = 1.5, tile: int = 8) -> np.ndarray:
    """CLAHE（对比度受限直方图均衡）。

    放在缓存阶段做（原来是训练时在线做），好处是每张图只算一次，
    数据加载端不再有这块 CPU 开销。
    clip 从 2.0 降到 1.5：目检发现 2.0 对侧位片过度增强出现斑驳伪影。
    """
    import cv2
    return cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(tile, tile)).apply(img)


def preprocess(img: np.ndarray, size: int, crop: bool = False,
               percentile: bool = False, clahe_clip: float = 0.0) -> np.ndarray:
    """完整的预处理链。所有开关默认关闭 = 与原有行为完全一致。

    顺序（与外部方案一致，但去掉了 1024 分辨率和 JPG 缓存）：
        原图 → [黑边裁剪] → [百分位截断] → [CLAHE] → letterbox 到 size
    """
    if crop:
        img = crop_black_border(img)
    if percentile:
        img = normalize_contrast(img)
    if clahe_clip > 0:
        img = apply_clahe(img, clip=clahe_clip)
    return letterbox(img, size)


def cache_paths(cache_dir: str, split: str, size: int) -> dict:
    """统一约定缓存文件名，避免各处手写字符串拼错。"""
    return {
        "array": os.path.join(cache_dir, f"{split}_{size}.npy"),
        "index": os.path.join(cache_dir, f"index_{split}.csv"),
        "labels": os.path.join(cache_dir, f"{split}_labels.npy"),
    }


# 多进程写入
def _fill_chunk(job: tuple) -> tuple[int, list[str]]:
    """worker：打开 memmap 并把自己负责的行写进去，返回 (成功数, 失败信息)。

    Windows 上是 spawn 启动，每个进程独立 `open_memmap(mode='r+')`，
    写入互不重叠的行，因此不需要加锁，也避免了把大数组通过管道回传
    （回传一整个 448×448 数组的 IPC 开销会把多进程的优势吃光）。

    job 里带 preprocessing 参数（crop / percentile / clahe_clip），
    全部默认关闭时行为与旧版完全一致。
    """
    arr_path, shape, size, entries, preprocessing_options = job
    crop, percentile, clahe_clip = preprocessing_options
    arr = np.lib.format.open_memmap(arr_path, mode="r+", dtype=np.uint8, shape=shape)
    failed: list[str] = []
    written = 0
    for row_idx, abspath in entries:
        try:
            img = load_gray(abspath)
            arr[row_idx] = preprocess(img, size, crop=crop, percentile=percentile,
                                      clahe_clip=clahe_clip)
            written += 1
        except Exception as exc:  # noqa: BLE001 - 单张坏图不该中断全量
            failed.append(f"{abspath}\t{type(exc).__name__}: {exc}")
    arr.flush()     # 显式落盘，进程被 kill 时已完成的部分不会丢
    del arr
    return written, failed


def build_array(df, data_root: str, arr_path: str, size: int,
                workers: int = 6, chunk_size: int = 64,
                crop: bool = False, percentile: bool = False,
                clahe_clip: float = 0.0) -> list[str]:
    """预分配 memmap 并并行填充。返回失败的路径列表。

    先一次性预分配整个文件（稀疏文件，不实际写满），再让各 worker 填各自的分块。
    这样避免了"边写边扩容"的内存峰值，也让进度可预测。
    """
    n = len(df)
    if os.path.exists(arr_path):
        raise FileExistsError(f"{arr_path} 已存在，请先手动删除或换个 size")

    # 预分配：mode="w+" 建文件并写 numpy 头（形状/dtype）
    np.lib.format.open_memmap(arr_path, mode="w+", dtype=np.uint8, shape=(n, size, size))
    arr = np.lib.format.open_memmap(arr_path, mode="r+")
    arr.flush()
    del arr

    # (行号, 绝对路径) 列表：行号来自 df.index，与 CSV 的顺序一致
    entries = [(i, os.path.join(data_root, rel),)
               for i, rel in zip(df.index.to_numpy(), df["relpath"])]
    image_chunks = [entries[i:i + chunk_size] for i in range(0, len(entries), chunk_size)]
    preprocessing_options = (bool(crop), bool(percentile), float(clahe_clip))
    cache_jobs = [(arr_path, (n, size, size), size, ch, preprocessing_options) for ch in image_chunks]

    print(f"  分配 {len(cache_jobs)} 个分块 ...", flush=True)
    failed: list[str] = []
    done = 0
    t0 = time.time()

    def _progress():
        """打印吞吐与预计剩余时间。done 用 nonlocal 修改外层变量。"""
        nonlocal done
        el = time.time() - t0
        rate = done / el if el > 0 else 0
        eta = (n - done - len(failed)) / rate if rate > 0 else 0
        print(f"  {done + len(failed)}/{n}  {rate:5.1f} img/s  "
              f"eta {eta / 60:5.1f} min  失败 {len(failed)}", flush=True)

    if workers <= 1:
        # 顺序回退：某些受限环境（沙箱/容器）禁止 multiprocessing 需要的命名管道，
        # 此时也能跑通，只是慢一点。调试时也方便直接看到异常栈。
        print("  以顺序模式运行（--workers 1）", flush=True)
        for job in cache_jobs:
            written, bad = _fill_chunk(job)
            done += written
            failed.extend(bad)
            if len(failed) or done % (chunk_size * 10) < chunk_size:
                _progress()
    else:
        print(f"  {workers} 进程并行", flush=True)
        # spawn 在 Windows 上是唯一选项，也比 fork 更安全（不会继承父进程的 CUDA 状态）
        worker_context = get_context("spawn")
        with worker_context.Pool(processes=workers) as pool:
            # imap_unordered：谁先完成谁先返回，避免慢分块拖住整体进度条
            for written, bad in pool.imap_unordered(_fill_chunk, cache_jobs, chunksize=1):
                done += written
                failed.extend(bad)
                if len(failed) or done % (chunk_size * 10) < chunk_size:
                    _progress()
    return failed


def compute_norm(arr_path: str, n_total: int, n_sample: int = 2000,
                 seed: int = 0) -> dict:
    """抽样统计灰度 mean/std（含 letterbox 黑边，与训练时所见一致）。

    为什么包含黑边：训练时模型输入就是带黑边的图，归一化统计量必须与之一致，
    否则实际输入分布会与预期偏移。
    """
    rng = np.random.default_rng(seed)
    # 只抽样，不读全量：3 万张 × 448² 全读进内存会 OOM
    idx = rng.choice(n_total, size=min(n_sample, n_total), replace=False)
    arr = np.lib.format.open_memmap(arr_path, mode="r")
    total = 0.0
    total_sq = 0.0
    count = 0
    for i in idx:
        v = arr[int(i)].astype(np.float64) / 255.0     # 归一到 [0,1] 再统计
        total += v.sum()
        total_sq += (v ** 2).sum()
        count += v.size
    del arr
    mean = total / count
    # 用 E[x²] - E[x]² 求方差；clamp 到正数防止浮点误差导致负数开根号
    var = max(total_sq / count - mean ** 2, 1e-12)
    return {"mean": float(mean), "std": float(var ** 0.5)}


def build_split(split: str, data_root: str, cache_dir: str, size: int,
                workers: int, limit: int | None = None,
                crop: bool = False, percentile: bool = False,
                clahe_clip: float = 0.0) -> dict:
    """构建单个划分的全部缓存产物。"""
    paths = cache_paths(cache_dir, split, size)
    os.makedirs(cache_dir, exist_ok=True)

    need_labels = split != "test"     # 测试集 CSV 没有 Predict_class 列
    df = load_split_df(split, data_root, need_labels=need_labels)
    if limit is not None:
        df = df.head(limit).copy()     # 调试：只处理前 N 张
    df = df.reset_index(drop=True)
    df["row_idx"] = np.arange(len(df))  # 行号 = memmap 第一维下标，必须严格递增

    print(f"[{split}] {len(df)} 张图 → {paths['array']}", flush=True)
    failed = build_array(df, data_root, paths["array"], size, workers=workers,
                         crop=crop, percentile=percentile, clahe_clip=clahe_clip)
    if failed:
        print(f"  !! {len(failed)} 张图解码失败，示例：", flush=True)
        for line in failed[:10]:
            print("    ", line, flush=True)

    df[INDEX_COLUMNS].to_csv(paths["index"], index=False)
    if need_labels:
        np.save(paths["labels"], labels_matrix(df))
    return {"n": len(df), "failed": len(failed), "array": paths["array"]}


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="构建 CXR 训练缓存")
    p.add_argument("--data-root", required=True,
                   help="数据集根目录（含 {split}.csv 与 {split}/ 子目录）")
    p.add_argument("--cache-dir", default="cache")
    p.add_argument("--size", type=int, default=448)
    p.add_argument("--splits", default="train,valid,test")
    p.add_argument("--workers", type=int, default=6,
                   help="本机可用内存有限，建议 4~6；1 表示顺序模式")
    p.add_argument("--limit", type=int, default=None,
                   help="每个划分只处理前 N 张（调试用）")
    # 预处理开关（默认全关 = 与原行为完全一致）
    p.add_argument("--crop-border", action="store_true",
                   help="裁掉探测器黑边（保守规则，宁可少裁不误裁）")
    p.add_argument("--percentile", action="store_true",
                   help="1%%~99%% 百分位截断线性拉伸，先稳住动态范围")
    p.add_argument("--clahe-clip", type=float, default=0.0,
                   help="在缓存阶段做 CLAHE，指定 clip 值（如 1.5）。0=不做，"
                        "此时训练时仍会在线做 clip=2.0 的 CLAHE")
    args = p.parse_args(argv)

    if args.crop_border or args.percentile or args.clahe_clip > 0:
        print(f"启用预处理: crop={args.crop_border} percentile={args.percentile} "
              f"clahe_clip={args.clahe_clip}", flush=True)
        print(f"  ⚠ 若想和旧缓存对比，请用不同的 --cache-dir（如 cache_v2）", flush=True)

    stats_by_split: dict[str, dict] = {}
    for split in args.splits.split(","):
        split = split.strip()
        if not split:
            continue
        stats_by_split[split] = build_split(
            split, args.data_root, args.cache_dir, args.size, args.workers, args.limit,
            crop=args.crop_border, percentile=args.percentile,
            clahe_clip=args.clahe_clip)

    # 归一化统计量只来自训练集（用验证/测试集会引入信息泄漏）
    train_stats = stats_by_split.get("train")
    if train_stats is None:
        print("\n警告：本次未构建 train 划分，跳过 norm.json 生成", flush=True)
        return 0

    train_path = cache_paths(args.cache_dir, "train", args.size)["array"]
    # 注意用 train 自己的行数，而不是循环结束后残留的最后一个划分的行数
    norm = compute_norm(train_path, train_stats["n"])

    # 尺寸特征（log(Rows), log(Columns)）的均值方差，供 dataset 做 z-score
    import pandas as pd
    meta = pd.read_csv(os.path.join(args.cache_dir, "index_train.csv"))
    log_h = np.log(meta["Rows"].astype(float).clip(lower=1))     # clip 防止 log(0)
    log_w = np.log(meta["Columns"].astype(float).clip(lower=1))
    norm["size_mean"] = [float(log_h.mean()), float(log_w.mean())]
    norm["size_std"] = [float(log_h.std() or 1.0), float(log_w.std() or 1.0)]
    norm["image_size"] = args.size
    with open(os.path.join(args.cache_dir, "norm.json"), "w", encoding="utf-8") as f:
        json.dump(norm, f, indent=2)

    print("\n完成：")
    for split, s in stats_by_split.items():
        print(f"  {split}: {s['n']} 张，失败 {s['failed']}")
    print(f"  norm: {norm}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
