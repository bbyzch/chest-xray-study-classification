"""在训练集 OOF 上为「患者平滑」逐类选 α，并可在官方 valid 上复核。

患者平滑：p' = (1-α)·p_本检查 + α·mean(同一患者其他检查的 p)，见 src/aggregate.py。

为什么值得做：测试集 1414 个检查只来自 136 个患者（平均每人 10.4 个检查，
训练集只有 1.67），同一患者的其他检查对本检查标签有独立信息。

α 的选择只用训练集 OOF：对每个外层折 k，在其余 4 折的 OOF 上逐类网格搜索，
再用 fold k 评估（留一折，不自评）；最终 α 取 5 折所选值的中位数。
valid 只用于复核，不参与选择。

用法：
    python tools/fit_patient_smooth.py --runs "runs_base384_pre22k/*"
    python tools/fit_patient_smooth.py --runs "runs_base384_pre22k/*" \
        --valid-probs valid_probs.npz      # 由 predict.py --split valid --save-probs 产出

最后一行打印可直接粘贴给 predict.py 的 --patient-smooth 参数。
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.aggregate import patient_smooth  # noqa: E402
from src.metrics import macro_auc, roc_auc  # noqa: E402

GRID = np.round(np.arange(0.0, 0.61, 0.05), 2)


def load_oof(patterns: list[str]):
    """读取若干实验的 5 折 OOF，图像级 mean 聚合到检查级；多个实验等权平均。"""
    per_exp = []
    for pat in patterns:
        files = sorted(glob.glob(os.path.join(pat, "fold*", "oof.npz")))
        if not files:
            raise FileNotFoundError(f"{pat} 下没有 fold*/oof.npz")
        for exp_dir in sorted({os.path.dirname(os.path.dirname(f)) for f in files}):
            P, Y, F = [], [], []
            for f in sorted(glob.glob(os.path.join(exp_dir, "fold*", "oof.npz"))):
                z = np.load(f)
                fold = int(os.path.basename(os.path.dirname(f))[4:])
                sid = z["study_id"].astype(str)
                p = pd.DataFrame(z["probs"]).groupby(sid).mean()
                y = pd.DataFrame(z["y"]).groupby(sid).first().loc[p.index]
                P.append(p); Y.append(y); F.append(pd.Series(fold, index=p.index))
            per_exp.append((exp_dir, pd.concat(P), pd.concat(Y), pd.concat(F)))
            print(f"  {exp_dir}: {len(per_exp[-1][1])} 个检查")
    common = per_exp[0][1].index
    for _, p, _, _ in per_exp[1:]:
        common = common.intersection(p.index)
    if len(common) < len(per_exp[0][1].index):
        print(f"  ⚠ 实验间检查不一致，取交集 {len(common)} 个")
    probs = np.mean([p.loc[common].values for _, p, _, _ in per_exp], axis=0)
    y = per_exp[0][2].loc[common].values.astype(int)
    fold = per_exp[0][3].loc[common].values
    return common.to_numpy(), probs, y, fold


def fit_lofo(probs, y, fold, subjects):
    """逐类留一折选 α。返回 (每折所选 α 矩阵, LOFO 平滑后的概率)。"""
    folds = np.unique(fold)
    chosen = np.zeros((len(folds), probs.shape[1]))
    out = probs.copy()
    smoothed = {a: patient_smooth(probs, subjects, a) for a in GRID}
    for i, k in enumerate(folds):
        tr, te = fold != k, fold == k
        for c in range(probs.shape[1]):
            best = max(GRID, key=lambda a: roc_auc(y[tr, c], smoothed[a][tr, c]))
            chosen[i, c] = best
            out[te, c] = smoothed[best][te, c]
    return chosen, out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="实验目录（支持通配符）")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--valid-probs", default=None,
                    help="predict.py --split valid --save-probs 产出的 npz，用于复核")
    args = ap.parse_args(argv)

    idx = pd.read_csv(os.path.join(args.cache_dir, "index_train.csv"))
    st2sub = idx.groupby("Study_id")["Subject_id"].first()

    print("读取 OOF：")
    sids, probs, y, fold = load_oof(args.runs)
    subjects = st2sub.loc[sids].to_numpy()

    chosen, lofo = fit_lofo(probs, y, fold, subjects)
    alphas = np.median(chosen, axis=0)
    base, base_pc = macro_auc(y, probs)
    lofo_auc, _ = macro_auc(y, lofo)
    print(f"\n训练 OOF：基线 {base:.4f}   留一折逐类平滑 {lofo_auc:.4f}  ({lofo_auc - base:+.4f})")
    print("各折所选 α（行=折，列=类别 1..10）：")
    print(np.array2string(chosen, precision=2))
    print("最终 α（中位数）:", alphas.tolist())

    if args.valid_probs:
        z = np.load(args.valid_probs)
        vmeta = pd.read_csv(os.path.join(args.cache_dir, "index_valid.csv"))
        vlab = np.load(os.path.join(args.cache_dir, "valid_labels.npy"))
        sid = z["study_id"].astype(str)
        vp = pd.DataFrame(z["probs"]).groupby(sid).mean()
        vy = pd.DataFrame(vlab).groupby(vmeta["Study_id"].to_numpy()).first().loc[vp.index]
        vsub = vmeta.groupby("Study_id")["Subject_id"].first().loc[vp.index].to_numpy()
        vb, vpc = macro_auc(vy.values, vp.values)
        vs, vspc = macro_auc(vy.values, patient_smooth(vp.values, vsub, alphas))
        print(f"\n官方 valid 复核（{len(vp)} 检查 / {len(np.unique(vsub))} 患者）："
              f"基线 {vb:.4f} → 平滑 {vs:.4f}  ({vs - vb:+.4f})")
        print("逐类差值:", np.round([vspc[c] - vpc[c] for c in range(10)], 4).tolist())

    spec = ",".join(f"{a:g}" for a in alphas)
    print(f"\n--patient-smooth \"{spec}\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
