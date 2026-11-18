"""用全量训练图像训练一个蒸馏学生模型。

教师软目标必须由 ``build_distill_targets.py`` 从训练集 OOF 生成，
因此这里不会读取测试图，也不会做验证折 checkpoint 选择。这个脚本是
“全量训练最终学生”的流程；若要评估蒸馏学生的五折泛化，不能直接复用
该 OOF 目标，必须使用 ``build_nested_distill_targets.py`` 按外层折生成目标。
训练轮数应根据已完成的五折实验固定后再运行，默认 12 轮，最终保存 EMA 权重。

示例：
    python tools/train_full_distill.py --arch convnext_base --image-size 384 \
      --epochs 12 --batch-size 24 --num-workers 8 --backbone-norm \
      --pretrained-source imagenet22k \
      --distill-targets cache/train_teacher_probs_weighted.npz \
      --runs-dir runs_base384_pre22k_distill_full
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import TrainConfig  # noqa: E402
from src.losses import (EMA, amp_dtype, bce_multilabel, cosine_lr,
                        distill_multilabel)  # noqa: E402
from src.models import build_model  # noqa: E402
from src.train import (load_cache_index, load_distill_targets, make_dataset,
                       sample_weights)  # noqa: E402
from src.labels import VIEW_VOCAB  # noqa: E402
from src.dataset import CXRMemmapDataset  # noqa: E402


class _OffsetRows(torch.utils.data.Dataset):
    """把子数据集的 row_idx 整体平移，避免 valid 行号与 train 行号冲突。

    蒸馏时用 row_idx 查教师软目标：train 占 0..N-1，valid 接在后面 N..N+M-1。
    放在模块顶层，Windows 的 spawn 子进程才能 pickle。
    """

    def __init__(self, ds, offset: int):
        self.ds = ds
        self.offset = int(offset)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        item = self.ds[i]
        item["row_idx"] = item["row_idx"] + self.offset
        return item


def load_valid_teacher(path: str, n_rows: int) -> np.ndarray:
    """读取 valid 的图像级教师概率（src.predict --ckpts ... --split valid --save-probs 的产物）。

    这些教师全部只在 train 上训练过，从未见过 valid，因此不构成泄漏。
    """
    d = np.load(path, allow_pickle=False)
    probs = d["probs"].astype(np.float32)
    if probs.shape != (n_rows, 10):
        raise ValueError(f"valid 教师概率应为 ({n_rows}, 10)，得到 {probs.shape}。"
                         "注意要用 --ckpts 模式的 --save-probs（图像级），不是 --deploy 模式（检查级）")
    return np.clip(probs, 1e-5, 1.0 - 1e-5)


def train_full(cfg: TrainConfig, stop_at: int | None = None,
               include_valid: bool = False, valid_teacher: str | None = None) -> str:
    init_seed = cfg.seed if cfg.init_seed is None else cfg.init_seed
    torch.manual_seed(init_seed)
    np.random.seed(init_seed)
    device = torch.device(cfg.device)

    meta, labels, norm = load_cache_index(cfg)
    teacher_all = load_distill_targets(cfg.distill_targets, len(labels))
    stop_epoch = cfg.epochs - 1 if stop_at is None else int(stop_at)
    if not 0 <= stop_epoch < cfg.epochs:
        raise ValueError(f"--stop-epoch 必须在 0..{cfg.epochs - 1}，得到 {stop_epoch}")

    ds = make_dataset(cfg, meta, labels, norm, train=True)
    if include_valid:
        # 规则第 3 条：valid 并入训练后不能再用于选择任何参数。
        # 这里的停止轮次、集成权重、平滑 α 都已在 train 上定好，不再看 valid。
        import pandas as pd
        vmeta = pd.read_csv(os.path.join(cfg.cache_dir, "index_valid.csv"))
        vlabels = np.load(os.path.join(cfg.cache_dir, "valid_labels.npy"))
        n_valid_full = len(vmeta)
        if not np.array_equal(vmeta["row_idx"].to_numpy(), np.arange(n_valid_full)):
            raise ValueError("index_valid.csv 的 row_idx 应为 0..M-1 顺序排列")
        if cfg.limit_images:
            vmeta = vmeta.head(cfg.limit_images).reset_index(drop=True)
            vlabels = vlabels[:cfg.limit_images]
        vds = CXRMemmapDataset(
            arr_path=os.path.join(cfg.cache_dir, f"valid_{cfg.cache_size}.npy"),
            rows=vmeta, image_size=cfg.image_size, train=True,
            mean=norm["mean"], std=norm["std"], labels=vlabels,
            use_clahe=cfg.use_clahe, size_mean=norm["size_mean"],
            size_std=norm["size_std"], cache_size=cfg.cache_size, aug=cfg.aug)
        if teacher_all is not None:
            if not valid_teacher:
                raise ValueError("蒸馏 + --include-valid 需要 --valid-teacher（valid 的教师概率）")
            vt = load_valid_teacher(valid_teacher, n_valid_full)[:len(vmeta)]
            teacher_all = np.concatenate([teacher_all, vt], axis=0)
        ds = torch.utils.data.ConcatDataset([ds, _OffsetRows(vds, len(labels))])
        print(f"并入官方 valid：train {len(labels)} + valid {len(vmeta)} = {len(ds)} 张图",
              flush=True)
    nw = cfg.num_workers
    dl = DataLoader(
        ds, batch_size=cfg.batch_size, shuffle=True, num_workers=nw,
        pin_memory=(device.type == "cuda"),
        drop_last=len(ds) > cfg.batch_size,
        persistent_workers=nw > 0,
    )

    model_kw = dict(
        num_view=len(VIEW_VOCAB), dropout=cfg.dropout,
        view_dim=cfg.view_dim, orient_dim=cfg.orient_dim,
        hidden=cfg.hidden, pool=cfg.pool, csra_lambda=cfg.csra_lambda,
        backbone_norm=cfg.backbone_norm,
        pretrained_source=cfg.pretrained_source,
        study_fusion=False,
        label_graph=cfg.label_graph,
        label_graph_alpha=cfg.label_graph_alpha,
        drop_path=cfg.drop_path,
    )
    model = build_model(cfg.arch, pretrained=cfg.pretrained, **model_kw).to(device)
    optimizer = torch.optim.AdamW(
        model.param_groups(cfg.lr, cfg.backbone_scale, cfg.weight_decay,
                           cfg.layer_decay))
    ema = EMA(model, cfg.ema_decay)

    ampf = amp_dtype(device) if cfg.amp else torch.float32
    use_scaler = ampf == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    steps_per_epoch = max(1, len(dl))
    total_steps = steps_per_epoch * cfg.epochs
    warmup = steps_per_epoch * cfg.warmup_epochs
    base_lrs = [g["lr"] for g in optimizer.param_groups]

    history = []
    step = 0
    print(f"device={device} arch={cfg.arch} size={cfg.image_size} "
          f"epochs={cfg.epochs} full_train={len(ds)}", flush=True)
    print(f"teacher={cfg.distill_targets or '无（普通监督）'} alpha={cfg.distill_alpha:g} "
          f"temperature={cfg.distill_temperature:g} stop_epoch={stop_epoch}", flush=True)

    for epoch in range(cfg.epochs):
        model.train()
        running = 0.0
        t0 = time.time()
        for batch in dl:
            scale = cosine_lr(step, total_steps, warmup, 1.0)
            for group, base in zip(optimizer.param_groups, base_lrs):
                group["lr"] = base * scale

            image = batch["image"].to(device, non_blocking=True)
            view = batch["view_idx"].to(device, non_blocking=True)
            orient = batch["orient_idx"].to(device, non_blocking=True)
            size = batch["size_feat"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            weight = sample_weights(view, cfg.lateral_weight)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=ampf,
                                    enabled=cfg.amp):
                logits = model(image, view, orient, size)
                hard = bce_multilabel(
                    logits, target, sample_weight=weight,
                    label_smoothing=cfg.label_smoothing)
                if teacher_all is None:
                    loss = hard
                else:
                    teacher = torch.from_numpy(
                        teacher_all[batch["row_idx"].numpy()]).to(device)
                    soft = distill_multilabel(
                        logits, teacher, temperature=cfg.distill_temperature,
                        sample_weight=weight)
                    loss = ((1.0 - cfg.distill_alpha) * hard
                            + cfg.distill_alpha * soft)

            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
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

        mean_loss = running / steps_per_epoch
        history.append({"epoch": epoch, "loss": mean_loss,
                        "sec": round(time.time() - t0, 1)})
        print(f"ep{epoch:02d} loss {mean_loss:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        if epoch == stop_epoch:
            break

    ema.copy_to(model)
    # 目录层级 <runs_dir>/<实验名>/full/ckpt.pt：predict.py 以 ckpt 上两级目录名作为
    # 集成分组名，这样 --weights 里写的就是实验名。
    tag = (cfg.exp_name + "_full" + ("_tv" if include_valid else "")
           + (f"_stop{stop_epoch}" if stop_at is not None else ""))
    out_dir = os.path.join(cfg.runs_dir, tag, "full")
    os.makedirs(out_dir, exist_ok=True)
    torch.save({
        "model": {k: v.detach().cpu().clone()
                  for k, v in model.state_dict().items()},
        "arch": cfg.arch,
        "image_size": cfg.image_size,
        "cache_size": cfg.cache_size,
        "cfg": cfg.to_json(),
        "best_auc": None,
        "best_epoch": stop_epoch,
    }, os.path.join(out_dir, "ckpt.pt"))
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"full_train": True, "history": history}, f,
                  ensure_ascii=False, indent=2)
    print(f"已保存全量单模型: {os.path.join(out_dir, 'ckpt.pt')}", flush=True)
    return out_dir


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--arch", default="convnext_base")
    p.add_argument("--image-size", type=int, default=384)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--pretrained-source",
                   choices=["torchvision", "imagenet22k", "clip_laion", "ssl_cxr"],
                   default="torchvision")
    p.add_argument("--ssl-weights", default=None,
                   help="--pretrained-source ssl_cxr 时的权重文件路径")
    p.add_argument("--backbone-norm", action="store_true")
    p.add_argument("--cache-dir", default="cache")
    p.add_argument("--cache-size", type=int, default=448)
    p.add_argument("--limit-images", type=int, default=None,
                   help="仅用于 smoke，正式全量训练不要设置")
    p.add_argument("--runs-dir", default="runs_full")
    p.add_argument("--distill-targets", default=None,
                   help="教师软目标；不给则为普通监督全量训练")
    p.add_argument("--distill-alpha", type=float, default=0.3)
    p.add_argument("--distill-temperature", type=float, default=2.0)
    p.add_argument("--stop-epoch", type=int, default=None,
                   help="在第 N 轮（0-based）结束后保存并停止；学习率仍按 --epochs 的完整 "
                        "cosine 调度走，与五折实验同一条曲线。由五折的最佳轮次决定，不看测试集")
    p.add_argument("--no-clahe", action="store_true")
    p.add_argument("--init-seed", type=int, default=None)
    p.add_argument("--drop-path", type=float, default=None)
    p.add_argument("--layer-decay", type=float, default=1.0)
    p.add_argument("--include-valid", action="store_true",
                   help="把官方 valid 并入训练（之后 valid 不能再用于任何选择）")
    p.add_argument("--valid-teacher", default=None,
                   help="valid 的图像级教师概率 npz；蒸馏 + --include-valid 时必需")
    args = p.parse_args(argv)
    if not 0.0 <= args.distill_alpha <= 1.0:
        p.error("--distill-alpha 必须在 0 和 1 之间")
    if args.pretrained_source == "ssl_cxr":
        if not args.ssl_weights:
            p.error("--pretrained-source ssl_cxr 需要 --ssl-weights")
        os.environ["CXR_SSL_WEIGHTS"] = os.path.abspath(args.ssl_weights)
    cfg = TrainConfig(
        arch=args.arch, image_size=args.image_size, epochs=args.epochs,
        batch_size=args.batch_size, lr=args.lr, num_workers=args.num_workers,
        seed=args.seed, init_seed=args.init_seed, device=args.device,
        pretrained=not args.no_pretrained,
        pretrained_source=args.pretrained_source,
        backbone_norm=args.backbone_norm, use_clahe=not args.no_clahe,
        drop_path=args.drop_path, layer_decay=args.layer_decay,
        cache_dir=args.cache_dir, cache_size=args.cache_size,
        limit_images=args.limit_images,
        runs_dir=args.runs_dir, loss="bce",
        distill_targets=args.distill_targets,
        distill_alpha=args.distill_alpha,
        distill_temperature=args.distill_temperature,
        save_oof=False,
        extra={"ssl_weights": os.environ.get("CXR_SSL_WEIGHTS")}
        if args.pretrained_source == "ssl_cxr" else {},
    )
    train_full(cfg, stop_at=args.stop_epoch, include_valid=args.include_valid,
               valid_teacher=args.valid_teacher)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
