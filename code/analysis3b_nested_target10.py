#!/usr/bin/env python3
"""
分析3b：Target-10 嵌套特征选择重跑（修复 test-set leakage 问题）
======================================================
背景：
  原 Table VI-a（Target-10 within-dataset compression）使用的 Top-10 特征
  来自 shap_signatures_rq1.py 单次计算（seed=42 的固定 test-set 正例样本），
  然后在 analysis3_statistical_validation.py 的 30-seed 重复评估中被冻结复用。
  由于 30 个 seed（42-71）都是对同一底层数据集的反复重切分，signature 提取
  用到的 seed=42 test-set 样本会大量重新出现在其他 seed 的 train/test 划分中，
  构成 feature-selection 阶段的 test-set information reuse 风险。

修复方案（本脚本）：
  对每个 (semantic_class, target) 组合的每个 seed，独立完成：
    Train_seed → XGBoost(40 feat) → SHAP(仅用 train 正例) → Top10_seed → Test_seed 评估
  即特征选择只使用该 seed 自己的训练集，评估只使用该 seed 自己的测试集，
  两者互不重叠，彻底消除该条件下的 test-set leakage。

  范围：仅 Table VI-a（Target-10 条件），因为 SHAP-10（跨数据集迁移条件，Table VI-b）
  的签名来自完全独立的源数据集，本来就不存在此问题。

用法：
  python3.11 analysis3b_nested_target10.py \
      --data-dir ./processed/ --out ./results/analysis3b/ --n-seeds 30
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

DATASET_SHORT = {
    "NF-UNSW-NB15-v2": "UNSW", "NF-CSE-CIC-IDS2018-v2": "CIC",
    "NF-ToN-IoT-v2": "ToN", "NF-BoT-IoT-v2": "BoT",
}
DS_CSV = {
    "UNSW": "NF-UNSW-NB15-v2_processed.csv",
    "CIC":  "NF-CSE-CIC-IDS2018-v2_processed.csv",
    "ToN":  "NF-ToN-IoT-v2_processed.csv",
    "BoT":  "NF-BoT-IoT-v2_processed.csv",
}
SEMANTIC_MAPPING = {
    "DoS": {
        "UNSW": ["DoS"],
        "CIC":  ["DoS attacks-Hulk", "DoS attacks-GoldenEye",
                 "DoS attacks-SlowHTTPTest", "DoS attacks-Slowloris"],
        "ToN":  ["dos"],
        "BoT":  ["DoS"],
    },
    "Reconnaissance": {
        "UNSW": ["Reconnaissance"],
        "ToN":  ["scanning"],
        "BoT":  ["Reconnaissance"],
    },
}
# Table VI-a 顺序
TARGET_ORDER = [("DoS", "UNSW"), ("DoS", "CIC"), ("DoS", "ToN"), ("DoS", "BoT"),
                ("Reconnaissance", "UNSW"), ("Reconnaissance", "ToN"), ("Reconnaissance", "BoT")]

NON_FEATURE_COLS = {"Label", "Attack"}

XGB_PARAMS = {
    "objective": "binary:logistic", "n_estimators": 300, "max_depth": 6,
    "learning_rate": 0.1, "subsample": 0.8, "colsample_bytree": 0.8,
    "eval_metric": "logloss", "n_jobs": -1, "tree_method": "hist",
    "missing": np.nan,
}
TOP_K = 10
TEST_SIZE = 0.20
MAX_TRAIN_POS = 30_000
NEG_RATIO_CAP = 10
SHAP_POS_CAP = 3_000   # SHAP 只在 train 正例上算，采样上限（控制耗时）


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out", default="./results/analysis3b/")
    p.add_argument("--classes", nargs="+", default=list(SEMANTIC_MAPPING.keys()))
    p.add_argument("--n-seeds", type=int, default=30)
    p.add_argument("--targets", nargs="+", default=None,
                   help="限定 target 数据集简称（调试用）")
    return p.parse_args()


def load_dataset(data_dir: Path, ds: str) -> pd.DataFrame:
    return pd.read_csv(data_dir / DS_CSV[ds], low_memory=False)


def clean_X(X: np.ndarray) -> np.ndarray:
    X[~np.isfinite(X)] = 0.0
    large = np.abs(X) > 1e15
    if large.any():
        for ci in np.where(large.any(axis=0))[0]:
            col = X[:, ci]
            valid = col[~large[:, ci]]
            col[large[:, ci]] = float(np.median(valid)) if len(valid) > 0 else 0.0
    return X


def build_full40(df: pd.DataFrame, feature_cols: list, target_labels: list, seed: int):
    pos_idx = df.index[df["Attack"].isin(target_labels)].to_numpy()
    neg_idx = df.index[~df["Attack"].isin(target_labels)].to_numpy()
    if len(pos_idx) == 0:
        return None
    rng = np.random.default_rng(seed)
    if len(pos_idx) > MAX_TRAIN_POS:
        pos_idx = rng.choice(pos_idx, MAX_TRAIN_POS, replace=False)
    neg_cap = len(pos_idx) * NEG_RATIO_CAP
    if len(neg_idx) > neg_cap:
        neg_idx = rng.choice(neg_idx, int(neg_cap), replace=False)
    X = clean_X(np.vstack([df.loc[pos_idx, feature_cols].to_numpy(dtype=np.float64),
                            df.loc[neg_idx, feature_cols].to_numpy(dtype=np.float64)]))
    y = np.concatenate([np.ones(len(pos_idx), dtype=np.int32),
                         np.zeros(len(neg_idx), dtype=np.int32)])
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=seed, stratify=y)


def fit_xgb(X_tr, y_tr, seed):
    params = XGB_PARAMS.copy()
    params["scale_pos_weight"] = int((y_tr == 0).sum()) / max(int((y_tr == 1).sum()), 1)
    params["random_state"] = seed
    m = xgb.XGBClassifier(**params)
    m.fit(X_tr, y_tr, verbose=False)
    return m


def evaluate(m, X_te, y_te):
    y_pred = m.predict(X_te)
    y_prob = m.predict_proba(X_te)[:, 1]
    return {"f1": float(f1_score(y_te, y_pred, zero_division=0)),
            "auc_roc": float(roc_auc_score(y_te, y_prob))}


def nested_top10_round(X_train, X_test, y_train, y_test, feature_cols, seed):
    """一次嵌套 Target-10 计算：train 内部完成 SHAP 特征选择，test 上评估。"""
    m_full = fit_xgb(X_train, y_train, seed)

    rng = np.random.default_rng(seed + 700000)
    pos_idx = np.where(y_train == 1)[0]
    if len(pos_idx) > SHAP_POS_CAP:
        pos_idx = rng.choice(pos_idx, SHAP_POS_CAP, replace=False)

    explainer = shap.TreeExplainer(m_full, feature_perturbation="tree_path_dependent")
    shap_values = explainer.shap_values(X_train[pos_idx])
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]
    abs_mean_shap = np.abs(shap_values).mean(axis=0)
    top10_idx = np.argsort(abs_mean_shap)[::-1][:TOP_K]
    top10_feats = [feature_cols[i] for i in top10_idx]

    Xtr10 = X_train[:, top10_idx]
    Xte10 = X_test[:, top10_idx]
    m10 = fit_xgb(Xtr10, y_train, seed)
    metrics = evaluate(m10, Xte10, y_test)
    metrics["top10_features"] = ";".join(top10_feats)
    return metrics


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_seeds = args.n_seeds

    all_feature_cols = None
    df_cache = {}

    rows = []
    for sem_cls, tgt_ds in TARGET_ORDER:
        if sem_cls not in args.classes:
            continue
        if args.targets and tgt_ds not in args.targets:
            continue
        if tgt_ds not in df_cache:
            df_cache[tgt_ds] = load_dataset(data_dir, tgt_ds)
        df = df_cache[tgt_ds]
        if all_feature_cols is None:
            all_feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
        tgt_labels = SEMANTIC_MAPPING[sem_cls][tgt_ds]

        print(f"\n== {sem_cls} / {tgt_ds} ({len(df):,d} rows) ==")
        for seed_off in range(n_seeds):
            seed = 42 + seed_off
            split = build_full40(df, all_feature_cols, tgt_labels, seed)
            if split is None:
                continue
            X_train, X_test, y_train, y_test = split
            m = nested_top10_round(X_train, X_test, y_train, y_test, all_feature_cols, seed)
            rows.append(dict(semantic_class=sem_cls, target=tgt_ds, seed=seed,
                              condition="Target-10-Nested",
                              f1=round(m["f1"], 6), auc_roc=round(m["auc_roc"], 6),
                              top10_features=m["top10_features"]))
            print(f"  seed {seed}: F1={m['f1']:.4f}  AUC={m['auc_roc']:.4f}")

    raw_df = pd.DataFrame(rows)
    raw_csv = out_dir / "analysis3b_nested_raw.csv"
    raw_df.to_csv(raw_csv, index=False)
    print(f"\n原始结果: {raw_csv} ({len(raw_df)} 行)")

    # 汇总 + 与原 frozen Target-10 对比（若可用）
    summary_rows = []
    frozen_path = Path("./results/analysis3/analysis3_raw_results.csv")
    frozen_df = pd.read_csv(frozen_path) if frozen_path.exists() else None

    for sem_cls, tgt_ds in TARGET_ORDER:
        sub = raw_df[(raw_df["semantic_class"] == sem_cls) & (raw_df["target"] == tgt_ds)]
        if sub.empty:
            continue
        nested_f1 = sub["f1"].values
        entry = dict(semantic_class=sem_cls, target=tgt_ds,
                     n_seeds=len(nested_f1),
                     nested_f1_mean=float(nested_f1.mean()),
                     nested_f1_std=float(nested_f1.std(ddof=1)))
        if frozen_df is not None:
            frozen_sub = frozen_df[(frozen_df["semantic_class"] == sem_cls) &
                                    (frozen_df["target"] == tgt_ds) &
                                    (frozen_df["condition"] == "Target-10")]
            if not frozen_sub.empty:
                frozen_f1 = frozen_sub.sort_values("seed")["f1"].values
                entry["frozen_f1_mean"] = float(frozen_f1.mean())
                entry["frozen_f1_std"] = float(frozen_f1.std(ddof=1))
                entry["delta_nested_minus_frozen"] = entry["nested_f1_mean"] - entry["frozen_f1_mean"]
                if len(frozen_f1) == len(nested_f1):
                    order = sub.sort_values("seed")["f1"].values
                    diff = order - frozen_f1
                    if np.all(diff == 0):
                        entry["wilcoxon_p"] = 1.0
                    else:
                        try:
                            entry["wilcoxon_p"] = float(wilcoxon(order, frozen_f1).pvalue)
                        except Exception:
                            entry["wilcoxon_p"] = np.nan
        # Full-40 参照
        full_sub = frozen_df[(frozen_df["semantic_class"] == sem_cls) &
                              (frozen_df["target"] == tgt_ds) &
                              (frozen_df["condition"] == "Full-40")] if frozen_df is not None else None
        if full_sub is not None and not full_sub.empty:
            entry["full40_f1_mean"] = float(full_sub["f1"].mean())
            entry["full40_f1_std"] = float(full_sub["f1"].std(ddof=1))
        summary_rows.append(entry)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = out_dir / "analysis3b_nested_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"汇总对比: {summary_csv}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
