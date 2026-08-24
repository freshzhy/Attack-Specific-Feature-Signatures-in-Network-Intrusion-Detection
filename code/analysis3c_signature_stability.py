#!/usr/bin/env python3
"""
分析3c（v1）：within-dataset SHAP signature 的跨-seed稳定性
======================================================
背景：
  RQ1/RQ2 报告的 attack-specific SHAP signature 目前均基于单次 seed=42 的
  train/test split 计算。这里量化换一个随机划分种子后，Top-10 signature
  的稳定程度，以及 signature 本身的跨-seed变异幅度与跨数据集差异相比谁大谁小。

范围：
  限定在本文已具备 30-seed 重复验证基础设施的两个攻击类别
  （DoS、Reconnaissance，4 个数据集共 7 个 semantic_class×target 组合），
  与论文 5.4(1) 对"safely compress"结论的既有 scoping 保持一致。
  全部 29 个攻击类别的稳定性分析留作未来工作。

  注：已被 analysis3c_signature_stability_v2.py 取代（该版本在每个 seed
  自己的 test-split positive samples 上计算 SHAP，与正式 signature 定义
  完全对齐），本脚本仅保留作历史参考。

方法：
  复用 analysis3b_nested_target10.py 完全相同的 nested 设计
  （每个 seed 独立完成 train-only SHAP 计算 → Top-10），但这次额外保存
  完整 40 维 |SHAP| 均值向量（而不仅仅是 Top-10 特征名），用于：
    (a) Top-10 Jaccard 相似度：同一 (class, target) 内所有 C(30,2) 个
        seed pair 的 Top-10 特征集合 Jaccard 相似度（均值 / 中位数）
    (b) 完整 40 特征 Spearman 秩相关：同样对所有 seed pair 计算
    (c) Dominant-feature（|SHAP|排名第一特征）跨 seed 一致性：
        30 个 seed 中，众数特征出现的比例

用法：
  python3.11 analysis3c_signature_stability.py \
      --data-dir ./processed/ --out ./results/analysis3c/ --n-seeds 30
"""

import argparse
import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import analysis3b_nested_target10 as base

warnings.filterwarnings("ignore")

TARGET_ORDER = base.TARGET_ORDER


def nested_round_with_full_vector(X_train, X_test, y_train, y_test, feature_cols, seed):
    """与 base.nested_top10_round 相同的 nested 设计，额外返回完整 40 维 SHAP 向量。"""
    import shap

    m_full = base.fit_xgb(X_train, y_train, seed)

    rng = np.random.default_rng(seed + 700000)
    pos_idx = np.where(y_train == 1)[0]
    if len(pos_idx) > base.SHAP_POS_CAP:
        pos_idx = rng.choice(pos_idx, base.SHAP_POS_CAP, replace=False)

    explainer = shap.TreeExplainer(m_full, feature_perturbation="tree_path_dependent")
    shap_values = explainer.shap_values(X_train[pos_idx])
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]
    abs_mean_shap = np.abs(shap_values).mean(axis=0)  # full 40-dim vector

    top10_idx = np.argsort(abs_mean_shap)[::-1][: base.TOP_K]
    top10_feats = [feature_cols[i] for i in top10_idx]

    Xtr10 = X_train[:, top10_idx]
    Xte10 = X_test[:, top10_idx]
    m10 = base.fit_xgb(Xtr10, y_train, seed)
    metrics = base.evaluate(m10, Xte10, y_test)
    metrics["top10_features"] = ";".join(top10_feats)
    metrics["shap_vector"] = abs_mean_shap
    return metrics


def jaccard(set_a, set_b):
    a, b = set(set_a), set(set_b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="./results/analysis3c/")
    ap.add_argument("--n-seeds", type=int, default=30)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_seeds = args.n_seeds

    all_feature_cols = None
    df_cache = {}
    per_config_vectors = {}   # (cls,tgt) -> {seed: shap_vector}
    per_config_top10 = {}     # (cls,tgt) -> {seed: [feat,...]}
    raw_rows = []

    for sem_cls, tgt_ds in TARGET_ORDER:
        if tgt_ds not in df_cache:
            df_cache[tgt_ds] = base.load_dataset(data_dir, tgt_ds)
        df = df_cache[tgt_ds]
        if all_feature_cols is None:
            all_feature_cols = [c for c in df.columns if c not in base.NON_FEATURE_COLS]
        tgt_labels = base.SEMANTIC_MAPPING[sem_cls][tgt_ds]

        print(f"\n== {sem_cls} / {tgt_ds} ({len(df):,d} rows) ==")
        vecs, tops = {}, {}
        for seed_off in range(n_seeds):
            seed = 42 + seed_off
            split = base.build_full40(df, all_feature_cols, tgt_labels, seed)
            if split is None:
                continue
            X_train, X_test, y_train, y_test = split
            m = nested_round_with_full_vector(X_train, X_test, y_train, y_test, all_feature_cols, seed)
            vecs[seed] = m["shap_vector"]
            tops[seed] = m["top10_features"].split(";")
            raw_rows.append(dict(semantic_class=sem_cls, target=tgt_ds, seed=seed,
                                  f1=round(m["f1"], 6), auc_roc=round(m["auc_roc"], 6),
                                  top10_features=m["top10_features"],
                                  dominant_feature=tops[seed][0]))
            print(f"  seed {seed}: dominant={tops[seed][0]}")
        per_config_vectors[(sem_cls, tgt_ds)] = vecs
        per_config_top10[(sem_cls, tgt_ds)] = tops

    raw_df = pd.DataFrame(raw_rows)
    raw_csv = out_dir / "analysis3c_signature_stability_raw.csv"
    raw_df.to_csv(raw_csv, index=False)
    print(f"\n原始结果: {raw_csv} ({len(raw_df)} 行)")

    # ---- pairwise stability metrics per config ----
    summary_rows = []
    for sem_cls, tgt_ds in TARGET_ORDER:
        vecs = per_config_vectors[(sem_cls, tgt_ds)]
        tops = per_config_top10[(sem_cls, tgt_ds)]
        seeds = sorted(vecs.keys())
        if len(seeds) < 2:
            continue
        jaccs, rhos = [], []
        for s1, s2 in itertools.combinations(seeds, 2):
            jaccs.append(jaccard(tops[s1], tops[s2]))
            rho, _ = spearmanr(vecs[s1], vecs[s2])
            rhos.append(rho)
        dominants = [tops[s][0] for s in seeds]
        mode_feat = pd.Series(dominants).value_counts().idxmax()
        mode_freq = pd.Series(dominants).value_counts().max() / len(dominants)

        summary_rows.append(dict(
            semantic_class=sem_cls, target=tgt_ds, n_seeds=len(seeds), n_pairs=len(jaccs),
            jaccard_mean=float(np.mean(jaccs)), jaccard_median=float(np.median(jaccs)),
            jaccard_min=float(np.min(jaccs)), jaccard_max=float(np.max(jaccs)),
            spearman_mean=float(np.mean(rhos)), spearman_median=float(np.median(rhos)),
            spearman_min=float(np.min(rhos)), spearman_max=float(np.max(rhos)),
            dominant_feature_mode=mode_feat, dominant_feature_freq=float(mode_freq),
        ))

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = out_dir / "analysis3c_signature_stability_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"稳定性汇总: {summary_csv}")
    print(summary_df.to_string(index=False))

    # ---- overall (pooled across 7 configs) ----
    overall = dict(
        jaccard_mean=summary_df["jaccard_mean"].mean(),
        jaccard_median=summary_df["jaccard_median"].median(),
        spearman_mean=summary_df["spearman_mean"].mean(),
        spearman_median=summary_df["spearman_median"].median(),
        dominant_feature_freq_mean=summary_df["dominant_feature_freq"].mean(),
        dominant_feature_freq_min=summary_df["dominant_feature_freq"].min(),
    )
    print("\n=== OVERALL (mean across 7 configs) ===")
    for k, v in overall.items():
        print(f"  {k}: {v:.4f}")
    pd.DataFrame([overall]).to_csv(out_dir / "analysis3c_overall.csv", index=False)


if __name__ == "__main__":
    main()
