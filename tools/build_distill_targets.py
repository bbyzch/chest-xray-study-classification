"""从训练集 OOF 预测构造图像级 OOF 多标签蒸馏软目标。

用法示例：
    python tools/build_distill_targets.py \
      --ensemble "runs/*/fold*" \
      --out cache/train_teacher_probs.npz

每个图像的目标只来自其所在验证折的 OOF 预测；同一折内多个教师实验
先等权平均，再写回全量 train row_idx。不会读取测试集，也不会生成测试伪标签。

注意：这种目标对“全量学生训练”是合适的，但不能直接作为另一个五折
学生的严格外层 CV 标签。后者必须按外层 fold 使用没有见过该验证折的
教师 checkpoint；请使用 ``build_nested_distill_targets.py``。
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np


def _fold_key(path: str) -> str:
    return os.path.basename(os.path.dirname(path))


def _experiment(path: str) -> str:
    parts = os.path.normpath(path).split(os.sep)
    return parts[-3] if len(parts) >= 3 else path


def _parse_weights(spec: str | None, paths: list[str]) -> dict[str, float]:
    names = sorted({_experiment(p) for p in paths})
    if not spec:
        return {name: 1.0 for name in names}
    out = {}
    for item in spec.split(","):
        name, value = item.rsplit(":", 1)
        out[name.strip()] = float(value)
    missing = [name for name in names if name not in out]
    if missing:
        raise ValueError(f"--weights 缺少教师实验: {missing}")
    if sum(out.values()) <= 0:
        raise ValueError("教师权重之和必须大于 0")
    return out


def build(paths: list[str], weight_spec: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    weights = _parse_weights(weight_spec, paths)
    experiment_folds: dict[str, set[str]] = {}
    for path in paths:
        parts = os.path.normpath(path).split(os.sep)
        if len(parts) < 3:
            continue
        experiment = parts[-3]
        experiment_folds.setdefault(experiment, set()).add(_fold_key(path))
    fold_sets = {tuple(sorted(v)) for v in experiment_folds.values()}
    if len(fold_sets) > 1:
        details = ", ".join(f"{k}:{sorted(v)}" for k, v in experiment_folds.items())
        raise ValueError("教师实验没有覆盖相同的 fold，不能安全混合：" + details)

    fold_groups: dict[str, list[str]] = {}
    for path in paths:
        fold_groups.setdefault(_fold_key(path), []).append(path)
    if len(fold_groups) < 2:
        raise ValueError("至少需要两个不同 fold 的 OOF 文件")

    all_rows, all_probs = [], []
    for fold, fold_paths in sorted(fold_groups.items()):
        rows_ref = None
        probs, prob_weights = [], []
        for path in sorted(fold_paths):
            d = np.load(path, allow_pickle=False)
            rows = d["row_idx"].astype(np.int64)
            p = d["probs"].astype(np.float32)
            if p.ndim != 2 or p.shape[1] != 10:
                raise ValueError(f"{path}: probs 应为 (N,10)，得到 {p.shape}")
            if rows_ref is None:
                rows_ref = rows
            elif not np.array_equal(rows_ref, rows):
                raise ValueError(f"{fold}: 教师 OOF 的 row_idx 不一致: {path}")
            probs.append(p)
            prob_weights.append(weights[_experiment(path)])
        all_rows.append(rows_ref)
        all_probs.append(np.average(np.stack(probs, axis=0), axis=0,
                                    weights=np.asarray(prob_weights)))
        print(f"{fold}: {len(rows_ref)} 张图，{len(probs)} 个教师")

    rows = np.concatenate(all_rows)
    probs = np.concatenate(all_probs).astype(np.float32)
    order = np.argsort(rows)
    rows, probs = rows[order], probs[order]
    if len(np.unique(rows)) != len(rows):
        raise ValueError("不同 fold 的 OOF row_idx 有重复，无法构造图像级 OOF 目标")
    if not np.array_equal(rows, np.arange(len(rows), dtype=np.int64)):
        raise ValueError(
            f"OOF 没有覆盖连续的 train row_idx 0..N-1，范围为 {rows.min()}..{rows.max()}，"
            f"实际数量 {len(rows)}")
    return rows, np.clip(probs, 1e-5, 1.0 - 1e-5)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ensemble", required=True, nargs="+",
                   help='一个或多个 OOF 通配符，如 "runs/*/fold*"')
    p.add_argument("--out", required=True, help="输出 npz 路径")
    p.add_argument("--weights", default=None,
                   help="实验权重，如 exp_a:0.7,exp_b:0.3；默认等权")
    args = p.parse_args(argv)
    paths = []
    for pattern in args.ensemble:
        paths.extend(glob.glob(os.path.join(pattern, "oof.npz")))
    paths = sorted(set(paths))
    if not paths:
        raise FileNotFoundError(f"没有找到 OOF: {args.ensemble}")
    rows, probs = build(paths, args.weights)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, row_idx=rows, teacher_probs=probs)
    print(f"已写出 {args.out}: {len(rows)} 张图，{probs.shape[1]} 类")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
