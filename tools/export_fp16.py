"""把 ckpt 的浮点权重存成 fp16（体积减半），并核对与 fp32 的输出差异。

规则第 8 条按"推理时实际加载的权重文件"计 500MB。4 个全量模型 fp32 约 886MB，
fp16 约 444MB。推理代码不用改：load_state_dict 会把 fp16 张量拷进 fp32 参数，
计算仍是 fp32。最终 submission.csv 与 cost.py 都必须从**同一份 fp16 文件**生成，
这样两边结果逐位一致，不依赖 |Δp|≤1e-3 的容差。

用法：
    python tools/export_fp16.py --src "runs_full/*/full" --out-root runs_full_fp16 --check 64
输出：runs_full_fp16/<实验名>/full/ckpt.pt（目录结构不变，predict.py 的 --weights 照用）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import CXRMemmapDataset  # noqa: E402
from src.predict import _infer_kwargs_from_cfg, expand_ckpts, infer, load_model  # noqa: E402


def to_fp16(ckpt: dict) -> dict:
    out = dict(ckpt)
    out["model"] = {k: (v.half() if v.dtype.is_floating_point else v)
                    for k, v in ckpt["model"].items()}
    out["weights_dtype"] = "float16"
    return out


def check(src_path: str, dst_path: str, args, device) -> float:
    meta = pd.read_csv(os.path.join(args.cache_dir, "index_valid.csv")).head(args.check)
    with open(os.path.join(args.cache_dir, "norm.json"), encoding="utf-8") as f:
        norm = json.load(f)
    outs = []
    for path in (src_path, dst_path):
        model, ck = load_model(path, device)
        ds = CXRMemmapDataset(
            arr_path=os.path.join(args.cache_dir, f"valid_{args.cache_size}.npy"),
            rows=meta, image_size=int(ck.get("image_size", 384)), train=False,
            mean=norm["mean"], std=norm["std"], labels=None,
            use_clahe=_infer_kwargs_from_cfg(ck)["use_clahe"],
            size_mean=norm["size_mean"], size_std=norm["size_std"],
            cache_size=args.cache_size)
        outs.append(infer(model, DataLoader(ds, batch_size=16), device))
        del model
    return float(np.abs(outs[0] - outs[1]).max())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", required=True, help="ckpt 目录或通配符")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--check", type=int, default=64, help="用前 N 张 valid 图核对，0=不核对")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--cache-size", type=int, default=448)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)
    device = torch.device(args.device)

    total = 0
    for src in expand_ckpts(args.src):
        parts = os.path.normpath(src).split(os.sep)
        dst = os.path.join(args.out_root, parts[-3], parts[-2], "ckpt.pt")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        torch.save(to_fp16(torch.load(src, map_location="cpu", weights_only=True)), dst)
        mb = os.path.getsize(dst) / 2**20
        total += os.path.getsize(dst)
        line = f"{parts[-3]:55s} {os.path.getsize(src) / 2**20:6.0f}MB -> {mb:5.0f}MB"
        if args.check:
            line += f"   max|Δp|={check(src, dst, args, device):.2e}"
        print(line, flush=True)
    print(f"合计 {total / 2**20:.0f}MB（规则上限 500MB，不含 config 等配套文件）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
