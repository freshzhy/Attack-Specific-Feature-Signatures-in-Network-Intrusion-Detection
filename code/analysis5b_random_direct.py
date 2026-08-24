#!/usr/bin/env python3
"""
分析5b：Random-10-Direct 零样本基线（补齐 SHAP-10-Direct 的对照）
======================================================
背景：
  analysis5_shap_direct_transfer.py 里的 "Random-10" 条件是目标数据集
  自训自测的下界基线，和 SHAP-10-Direct（源训目标零样本测）不是同口径比较，
  因此现有结果无法回答"SHAP 引导的源特征是否比随机源特征更适合零样本迁移"。

修复方案（本脚本）：
  新增 Random-10-Direct 条件：随机抽 10 个特征，在 source 数据集上训练，
  直接在 target 数据集的零样本测试集上评估（不重训练）。
  target 测试集构建逻辑与 SHAP-10-Direct 完全一致（同一 seed 对应同一批
  target 评估样本），确保 SHAP-10-Direct vs Random-10-Direct 可配对比较。

  复用 analysis5_shap_direct_transfer.py 里已验证过的 IO / 特征加载 /
  训练评估 / 统计检验函数，避免逻辑漂移。

用法：
  python3.11 analysis5b_random_direct.py \
      --rq1-dir ./results/rq1/ --data-dir ./processed/ \
      --out ./results/analysis5b/ --n-seeds 30
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import analysis5_shap_direct_transfer as base

warnings.filterwarnings("ignore")

TOP_K = base.TOP_K
TEST_SIZE = base.TEST_SIZE
MAX_TRAIN_POS = base.MAX_TRAIN_POS
NEG_RATIO_CAP = base.NEG_RATIO_CAP


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--rq1-dir", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out", default="./results/analysis5b/")
    p.add_argument("--classes", nargs="+", default=list(base.SEMANTIC_MAPPING.keys())
                   if hasattr(base, "SEMANTIC_MAPPING") else None)
    p.add_argument("--n-seeds", type=int, default=30)
    return p.parse_args()


def run_random_direct(sem_cls: str, mapping: dict, all_sigs: dict,
                       data_dir: Path, n_seeds: int) -> list[dict]:
    available = [ds for ds in base.DS_ORDER if ds in mapping and ds in all_sigs]

    feature_cols = None
    for ds in available:
        labels = mapping[ds]
        matched = {l: all_sigs[ds][l] for l in labels if l in all_sigs[ds]}
        if matched:
            feature_cols = list(matched.values())[0]["feature_cols"]
            break
    if feature_cols is None:
        return []

    print(f"\n  语义类别: {sem_cls}  (n_seeds={n_seeds})")
    datasets = {}
    for ds in available:
        df = base.load_df(data_dir, ds)
        if df is not None:
            datasets[ds] = df
            print(f"    {ds}: {len(df):,d} 行")

    rows = []
    seed_base = 42

    for tgt_ds in available:
        if tgt_ds not in datasets:
            continue
        tgt_df = datasets[tgt_ds]
        tgt_labels = mapping[tgt_ds]
        print(f"\n  ── 目标: {tgt_ds} ──")

        for seed_off in range(n_seeds):
            seed = seed_base + seed_off

            for src_ds in available:
                if src_ds == tgt_ds or src_ds not in datasets:
                    continue
                env_same = base.DS_ENV.get(src_ds) == base.DS_ENV.get(tgt_ds)

                # 每个 (src, tgt, seed) 独立抽一组随机 10 特征
                rng_feat = np.random.default_rng(seed * 1_000_003 +
                                                  abs(hash((src_ds, tgt_ds))) % 100_000)
                rand_feats = list(rng_feat.choice(feature_cols, size=TOP_K, replace=False))

                # 在 source 上训练（与 SHAP-10-Direct 相同的训练集构建方式）
                src_df = datasets[src_ds]
                res_src = base.build_ovr(src_df, rand_feats, mapping[src_ds], seed,
                                          test_size=TEST_SIZE)
                if res_src[0] is None:
                    continue
                m_direct = base.train_model(res_src[0], res_src[2], seed)

                # target 零样本测试集构建：与 SHAP-10-Direct 完全一致的抽样逻辑，
                # 确保同一 seed 下两条件评估于相同的 target 样本（可配对比较）
                pos_idx_tgt = tgt_df.index[tgt_df["Attack"].isin(tgt_labels)].to_numpy()
                neg_idx_tgt = tgt_df.index[~tgt_df["Attack"].isin(tgt_labels)].to_numpy()
                if len(pos_idx_tgt) == 0:
                    continue

                rng_tgt = np.random.default_rng(seed + 200000)
                n_pos_test = max(100, int(min(len(pos_idx_tgt), MAX_TRAIN_POS) * TEST_SIZE))
                n_neg_test = min(len(neg_idx_tgt), n_pos_test * NEG_RATIO_CAP)
                pos_test = rng_tgt.choice(pos_idx_tgt, n_pos_test, replace=False)
                neg_test = rng_tgt.choice(neg_idx_tgt, n_neg_test, replace=False)

                X_tgt_test = base.clean_X(np.vstack([
                    tgt_df.loc[pos_test, rand_feats].to_numpy(),
                    tgt_df.loc[neg_test, rand_feats].to_numpy(),
                ]))
                y_tgt_test = np.concatenate([
                    np.ones(len(pos_test), dtype=np.int32),
                    np.zeros(len(neg_test), dtype=np.int32),
                ])

                rows.append(dict(
                    semantic_class=sem_cls, target=tgt_ds,
                    source=src_ds, condition="Random-10-Direct",
                    env_same=env_same, seed=seed,
                    **base.evaluate(m_direct, X_tgt_test, y_tgt_test),
                ))

        print(f"    完成 {n_seeds} 轮")

    return rows


def main():
    args = parse_args()
    rq1_dir = Path(args.rq1_dir)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_seeds = args.n_seeds

    all_sigs = base.load_shap_sigs(rq1_dir)
    print(f"已加载 {len(all_sigs)} 个数据集的 SHAP 签名")

    classes = args.classes or list(base.SEMANTIC_MAPPING.keys())
    all_rows = []
    for sem_cls in classes:
        if sem_cls not in base.SEMANTIC_MAPPING:
            continue
        rows = run_random_direct(sem_cls, base.SEMANTIC_MAPPING[sem_cls],
                                  all_sigs, data_dir, n_seeds)
        all_rows.extend(rows)

    raw_df = pd.DataFrame(all_rows)
    raw_csv = out_dir / "analysis5b_random_direct_raw.csv"
    raw_df.to_csv(raw_csv, index=False)
    print(f"\n原始结果: {raw_csv} ({len(raw_df)} 行)")

    # 与已有 SHAP-10-Direct 结果做配对 Wilcoxon（按 semantic_class/source/target 分组，
    # 30 个 seed 配对）
    direct_path = Path("./results/analysis5/analysis5_raw_results.csv")
    if direct_path.exists():
        shap_direct = pd.read_csv(direct_path)
        shap_direct = shap_direct[shap_direct["condition"] == "SHAP-10-Direct"]
        summary_rows = []
        for (cls, src, tgt), grp in raw_df.groupby(["semantic_class", "source", "target"]):
            rand_f1 = grp.sort_values("seed").set_index("seed")["f1"]
            rand_auc = grp.sort_values("seed").set_index("seed")["auc_roc"]
            shap_sub = shap_direct[(shap_direct["semantic_class"] == cls) &
                                    (shap_direct["source"] == src) &
                                    (shap_direct["target"] == tgt)]
            if shap_sub.empty:
                continue
            shap_f1 = shap_sub.sort_values("seed").set_index("seed")["f1"]
            shap_auc = shap_sub.sort_values("seed").set_index("seed")["auc_roc"]
            common = rand_f1.index.intersection(shap_f1.index)
            if len(common) < 2:
                continue
            f1_diff = shap_f1.loc[common].values - rand_f1.loc[common].values
            try:
                if np.all(f1_diff == 0):
                    p_f1 = 1.0
                else:
                    p_f1 = float(wilcoxon(shap_f1.loc[common], rand_f1.loc[common],
                                          alternative="greater").pvalue)
            except Exception:
                p_f1 = np.nan
            summary_rows.append(dict(
                semantic_class=cls, source=src, target=tgt,
                n_seeds=len(common),
                random_direct_auc_mean=float(rand_auc.loc[common].mean()),
                random_direct_f1_mean=float(rand_f1.loc[common].mean()),
                shap_direct_auc_mean=float(shap_auc.loc[common].mean()),
                shap_direct_f1_mean=float(shap_f1.loc[common].mean()),
                delta_auc=float(shap_auc.loc[common].mean() - rand_auc.loc[common].mean()),
                wilcoxon_p_f1_shap_gt_random=p_f1,
            ))
        summary_df = pd.DataFrame(summary_rows)
        summary_csv = out_dir / "analysis5b_summary_vs_shap_direct.csv"
        summary_df.to_csv(summary_csv, index=False)
        print(f"与 SHAP-10-Direct 对比: {summary_csv}")
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
