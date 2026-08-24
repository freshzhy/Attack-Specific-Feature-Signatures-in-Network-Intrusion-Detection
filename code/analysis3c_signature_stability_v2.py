#!/usr/bin/env python3
"""
分析3c v2：within-dataset SHAP signature 的跨-seed稳定性（test-set positives 版本）
======================================================
背景：
  analysis3c（v1）的 30-seed 稳定性分析复用了 analysis3b 的 nested 设计，
  在每个 seed 的 **training-split positive samples** 上计算 SHAP 向量。
  但论文正式的 RQ1/RQ2 signature σ(c,D)（3.2.1 Eq.1）定义为：
  在 seed=42 的 **test-set positive samples** 上计算的 mean |SHAP| 向量
  （见 shap_signatures_rq1.py 第318行：
  `explainer.shap_values(X_test[pos_indices])`，以及论文 3.2.2 正文
  "computed over the test-set positive samples from a single fixed
  train/test split"）。

  即 v1 实际比较的是：
    reference signature（test-based, 单次 seed=42）
    vs.
    stability signatures（train-based, 30 seeds）
  这不是完全同一 extraction protocol，统计上不够严谨。

本脚本（v2）的改动：
  仅将 SHAP 向量的计算样本来源从 X_train[pos_idx] 改为 X_test[pos_idx]，
  与 shap_signatures_rq1.py 完全一致的 protocol（test-set positives），
  其余（30 seeds、nested per-seed 分类器训练、Top-10 Jaccard / 全40维
  Spearman / dominant-feature 频率的计算方式）保持不变。

  注：此处不存在 analysis3b/analysis3c v1 nested 设计原本要规避的
  leakage 风险 —— 那个风险specifically 是"用同一 test 集选出的 Top-10
  再拿同一 test 集去评估压缩后的性能"（double dipping）。本脚本只关心
  signature 本身（Top-10 特征集合 + 完整40维向量）的跨-seed稳定性，
  不用它来做后续同一 test 集上的性能评估，因此在 test positives 上
  计算 SHAP 是安全的，且与正式 signature 定义完全对齐。

用法：
  python3.11 analysis3c_signature_stability_v2.py \
      --data-dir ./processed/ --out ./results/analysis3c_v2/ --n-seeds 30
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


def nested_round_test_positives(X_train, X_test, y_train, y_test, feature_cols, seed):
    """与 v1 相同的 nested 分类器训练设计，但 SHAP 向量改为在
    test-set positive samples 上计算，与正式 signature 定义（3.2.1 Eq.1）
    protocol 完全一致。"""
    import shap

    m_full = base.fit_xgb(X_train, y_train, seed)

    rng = np.random.default_rng(seed + 800000)  # 与 v1 的 +700000 区分
    pos_idx = np.where(y_test == 1)[0]
    if len(pos_idx) > base.SHAP_POS_CAP:
        pos_idx = rng.choice(pos_idx, base.SHAP_POS_CAP, replace=False)

    explainer = shap.TreeExplainer(m_full, feature_perturbation="tree_path_dependent")
    shap_values = explainer.shap_values(X_test[pos_idx])
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]
    abs_mean_shap = np.abs(shap_values).mean(axis=0)  # full 40-dim vector

    top10_idx = np.argsort(abs_mean_shap)[::-1][: base.TOP_K]
    top10_feats = [feature_cols[i] for i in top10_idx]

    metrics = {}
    metrics["top10_features"] = ";".join(top10_feats)
    metrics["shap_vector"] = abs_mean_shap
    metrics["n_test_pos_used"] = int(len(pos_idx))
    return metrics


def jaccard(set_a, set_b):
    a, b = set(set_a), set(set_b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="./results/analysis3c_v2/")
    ap.add_argument("--n-seeds", type=int, default=30)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_seeds = args.n_seeds

    all_feature_cols = None
    df_cache = {}
    per_config_vectors = {}
    per_config_top10 = {}
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
            m = nested_round_test_positives(X_train, X_test, y_train, y_test, all_feature_cols, seed)
            vecs[seed] = m["shap_vector"]
            tops[seed] = m["top10_features"].split(";")
            raw_rows.append(dict(semantic_class=sem_cls, target=tgt_ds, seed=seed,
                                  n_test_pos_used=m["n_test_pos_used"],
                                  top10_features=m["top10_features"],
                                  dominant_feature=tops[seed][0]))
            print(f"  seed {seed}: dominant={tops[seed][0]} (n_test_pos={m['n_test_pos_used']})")
        per_config_vectors[(sem_cls, tgt_ds)] = vecs
        per_config_top10[(sem_cls, tgt_ds)] = tops

    raw_df = pd.DataFrame(raw_rows)
    raw_csv = out_dir / "analysis3c_v2_signature_stability_raw.csv"
    raw_df.to_csv(raw_csv, index=False)
    print(f"\n原始结果: {raw_csv} ({len(raw_df)} 行)")

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
    summary_csv = out_dir / "analysis3c_v2_signature_stability_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"稳定性汇总: {summary_csv}")
    print(summary_df.to_string(index=False))

    overall = dict(
        jaccard_mean=summary_df["jaccard_mean"].mean(),
        jaccard_median=summary_df["jaccard_median"].median(),
        spearman_mean=summary_df["spearman_mean"].mean(),
        spearman_median=summary_df["spearman_median"].median(),
        dominant_feature_freq_mean=summary_df["dominant_feature_freq"].mean(),
        dominant_feature_freq_min=summary_df["dominant_feature_freq"].min(),
    )
    print("\n=== OVERALL (mean across 7 configs, test-positive protocol) ===")
    for k, v in overall.items():
        print(f"  {k}: {v:.4f}")
    pd.DataFrame([overall]).to_csv(out_dir / "analysis3c_v2_overall.csv", index=False)


if __name__ == "__main__":
    main()
