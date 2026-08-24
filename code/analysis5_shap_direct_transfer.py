#!/usr/bin/env python3
"""
分析5：SHAP-10-Direct 跨数据集直接迁移实验
==========================================================
动机：
  原版 SHAP-10 在 target 数据集上重新训练，只是"特征集迁移"
  不是真正的 cross-dataset model transfer；需要补充一个真正的模型迁移条件。

新增条件 SHAP-10-Direct：
  - 用 source 数据集训练 XGBoost（限定 source Top-10 特征）
  - 直接在 target 测试集上评估，不做任何 target 训练
  - 这才是真正意义上的跨数据集迁移（zero-shot transfer）

实验设计（完整四条件对比）：
  Full-40       : 全特征，target 训练 + target 测试（性能上界）
  SHAP-10       : source Top-10，target 训练 + target 测试（特征集迁移）
  SHAP-10-Direct: source Top-10，source 训练 + target 测试（模型迁移）← 新增
  Random-10     : 随机10特征，target 训练 + target 测试（下界基线）

统计检验：
  SHAP-10-Direct vs SHAP-10  : 量化"模型迁移"vs"特征集迁移"的差距
  SHAP-10-Direct vs Random-10: 量化零样本迁移是否优于无信息基线
  使用 30 seeds（与分析3一致），Wilcoxon signed-rank + Holm-Bonferroni

输出：
  analysis5_raw_results.csv      — 原始 F1 数据
  analysis5_statistical_tests.csv — 统计检验（三种对比）
  analysis5_summary_table.csv    — 论文用汇总表
  analysis5_figure.pdf/png       — 主结果图（Fig. 11 候选）

用法：
  # 先验证（DoS，10 seeds）
  python analysis5_shap_direct_transfer.py \\
      --rq1-dir ./results/rq1/ \\
      --data-dir ./processed/ \\
      --out ./results/analysis5/ \\
      --classes DoS --n-seeds 10

  # 全量（30 seeds）
  python analysis5_shap_direct_transfer.py \\
      --rq1-dir ./results/rq1/ \\
      --data-dir ./processed/ \\
      --out ./results/analysis5/ \\
      --n-seeds 30
"""

import argparse
import json
import pickle
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import xgboost as xgb

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════════════════════════════
DATASET_SHORT = {
    "NF-UNSW-NB15-v2":       "UNSW",
    "NF-CSE-CIC-IDS2018-v2": "CIC",
    "NF-ToN-IoT-v2":         "ToN",
    "NF-BoT-IoT-v2":         "BoT",
}
DS_ORDER = ["UNSW", "CIC", "ToN", "BoT"]
DS_CSV = {
    "UNSW": "NF-UNSW-NB15-v2_processed.csv",
    "CIC":  "NF-CSE-CIC-IDS2018-v2_processed.csv",
    "ToN":  "NF-ToN-IoT-v2_processed.csv",
    "BoT":  "NF-BoT-IoT-v2_processed.csv",
}
DS_ENV = {"UNSW": "IT", "CIC": "IT", "ToN": "IoT", "BoT": "IoT"}

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

XGB_PARAMS = {
    "objective":        "binary:logistic",
    "n_estimators":     300,
    "max_depth":        6,
    "learning_rate":    0.1,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "eval_metric":      "logloss",
    "n_jobs":           -1,
    "tree_method":      "hist",
    "missing":          np.nan,
}

TOP_K          = 10
TEST_SIZE      = 0.20
MAX_TRAIN_POS  = 30_000
NEG_RATIO_CAP  = 10
ALPHA          = 0.05
BOOTSTRAP_N    = 1000

COND_COLORS = {
    "Full-40":         "#2166ac",
    "Target-10":       "#4dac26",
    "SHAP-10":         "#f4a582",
    "SHAP-10-Direct":  "#d6604d",
    "Random-10":       "#aaaaaa",
}
COND_ORDER = ["Full-40", "SHAP-10", "SHAP-10-Direct", "Random-10"]

STYLE = {"full_width": 7.16, "fig_dpi": 300, "font_size": 8}


def parse_args():
    p = argparse.ArgumentParser(description="分析5：SHAP-10-Direct 迁移实验")
    p.add_argument("--rq1-dir",  required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out",      default="./results/analysis5/")
    p.add_argument("--classes",  nargs="+",
                   default=list(SEMANTIC_MAPPING.keys()))
    p.add_argument("--n-seeds",  type=int, default=30)
    return p.parse_args()


def setup_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": STYLE["font_size"],
        "axes.titlesize": 9, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "savefig.dpi": STYLE["fig_dpi"], "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })


def save_fig(fig, stem: Path):
    fig.savefig(str(stem) + ".pdf", format="pdf")
    fig.savefig(str(stem) + ".png", format="png")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# 数据工具
# ══════════════════════════════════════════════════════════════════════════════
def load_shap_sigs(rq1_dir: Path) -> dict:
    sigs = {}
    for ds_full, ds_short in DATASET_SHORT.items():
        pkl = rq1_dir / f"{ds_full}_rq1_shap_values.pkl"
        if pkl.exists():
            with open(pkl, "rb") as f:
                sigs[ds_short] = pickle.load(f)
    return sigs


def get_topk(sigs: dict, raw_labels: list,
             feature_cols: list, k: int) -> list:
    matched = {l: sigs[l] for l in raw_labels if l in sigs}
    if not matched:
        return []
    total_w, weighted = 0, np.zeros(len(feature_cols))
    for sig in matched.values():
        w = sig.get("n_pos_shap", 1)
        weighted += sig["abs_mean_shap"] * w
        total_w  += w
    return [feature_cols[i]
            for i in np.argsort(weighted / total_w)[::-1][:k]]


def load_df(data_dir: Path, ds: str) -> pd.DataFrame | None:
    p = data_dir / DS_CSV[ds]
    return pd.read_csv(p, low_memory=False) if p.exists() else None


def clean_X(X: np.ndarray) -> np.ndarray:
    X = X.astype(np.float64)
    X[~np.isfinite(X)] = 0.0
    large = np.abs(X) > 1e15
    if large.any():
        for ci in np.where(large.any(axis=0))[0]:
            col = X[:, ci]
            valid = col[~large[:, ci]]
            col[large[:, ci]] = float(np.median(valid)) if len(valid) > 0 else 0.0
    return X


def build_ovr(df: pd.DataFrame, feat_cols: list,
              target_labels: list, seed: int,
              test_size: float = TEST_SIZE):
    """构建 OvR 数据集，返回 X_train, X_test, y_train, y_test"""
    pos_idx = df.index[df["Attack"].isin(target_labels)].to_numpy()
    neg_idx = df.index[~df["Attack"].isin(target_labels)].to_numpy()
    if len(pos_idx) == 0:
        return None, None, None, None
    rng = np.random.default_rng(seed)
    if len(pos_idx) > MAX_TRAIN_POS:
        pos_idx = rng.choice(pos_idx, MAX_TRAIN_POS, replace=False)
    neg_cap = len(pos_idx) * NEG_RATIO_CAP
    if len(neg_idx) > neg_cap:
        neg_idx = rng.choice(neg_idx, int(neg_cap), replace=False)
    X = clean_X(np.vstack([
        df.loc[pos_idx, feat_cols].to_numpy(),
        df.loc[neg_idx, feat_cols].to_numpy(),
    ]))
    y = np.concatenate([
        np.ones(len(pos_idx),  dtype=np.int32),
        np.zeros(len(neg_idx), dtype=np.int32),
    ])
    return train_test_split(X, y, test_size=test_size,
                            random_state=seed, stratify=y)


def train_model(X_tr: np.ndarray, y_tr: np.ndarray,
                seed: int) -> xgb.XGBClassifier:
    params = XGB_PARAMS.copy()
    params["scale_pos_weight"] = int((y_tr==0).sum()) / max(int((y_tr==1).sum()), 1)
    params["random_state"] = seed
    m = xgb.XGBClassifier(**params)
    m.fit(X_tr, np.asarray(y_tr, dtype=np.int32), verbose=False)
    return m


def evaluate(model: xgb.XGBClassifier,
             X_te: np.ndarray, y_te: np.ndarray) -> dict:
    y_te   = np.asarray(y_te, dtype=np.int32)
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]
    return {
        "f1":      round(float(f1_score(y_te, y_pred, zero_division=0)), 6),
        "auc_roc": round(float(roc_auc_score(y_te, y_prob)), 6),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 统计工具
# ══════════════════════════════════════════════════════════════════════════════
def bootstrap_ci(vals: np.ndarray, n: int = BOOTSTRAP_N,
                 seed: int = 0) -> tuple[float, float]:
    rng  = np.random.default_rng(seed)
    boot = [rng.choice(vals, len(vals), replace=True).mean() for _ in range(n)]
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    sp = np.sqrt(((na-1)*a.std(ddof=1)**2 + (nb-1)*b.std(ddof=1)**2) / (na+nb-2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else 0.0


def holm_bonferroni(pvals: list) -> list:
    n = len(pvals)
    idx_sorted = sorted(range(n), key=lambda i: pvals[i])
    corrected = [1.0] * n
    running_min = 1.0
    for rank, orig_i in enumerate(reversed(idx_sorted)):
        k = n - rank
        corrected[orig_i] = min(running_min, pvals[orig_i] * k)
        running_min = corrected[orig_i]
    return corrected


def wilcox_test(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    diff = a - b
    if np.all(diff == 0):
        return np.nan, 1.0
    try:
        res = wilcoxon(a, b, alternative="greater")
        return float(res.statistic), float(res.pvalue)
    except Exception:
        return np.nan, 1.0


# ══════════════════════════════════════════════════════════════════════════════
# 主实验
# ══════════════════════════════════════════════════════════════════════════════
def run_experiment(sem_cls: str, mapping: dict,
                   all_sigs: dict, data_dir: Path,
                   n_seeds: int) -> list[dict]:
    """
    对一个语义类别的所有 source-target 配对运行四条件实验。

    SHAP-10-Direct 的关键逻辑：
      1. 在 source 数据集上构建 OvR 训练集（source Top-10 特征）
      2. 训练 XGBoost 模型
      3. 在 target 数据集构建测试集（同样 source Top-10 特征）
      4. 直接用 source 模型预测 target 测试集
      注意：target 特征值分布与 source 不同，但特征名称相同（NFv2 统一空间）
    """
    print(f"\n  语义类别: {sem_cls}  (n_seeds={n_seeds})")
    print(f"  {'─'*58}")

    available = [ds for ds in DS_ORDER
                 if ds in mapping and ds in all_sigs]

    # 获取特征列
    feature_cols = None
    for ds in available:
        labels  = mapping[ds]
        matched = {l: all_sigs[ds][l] for l in labels if l in all_sigs[ds]}
        if matched:
            feature_cols = list(matched.values())[0]["feature_cols"]
            break
    if feature_cols is None:
        return []

    print("  加载数据集...")
    datasets = {}
    for ds in available:
        df = load_df(data_dir, ds)
        if df is not None:
            datasets[ds] = df
            print(f"    {ds}: {len(df):,d} 行")

    rows = []
    seed_base = 42

    for tgt_ds in available:
        if tgt_ds not in datasets:
            continue
        tgt_df     = datasets[tgt_ds]
        tgt_labels = mapping[tgt_ds]
        tgt_topk   = get_topk(all_sigs[tgt_ds], tgt_labels,
                               feature_cols, TOP_K)
        rng_rand   = np.random.default_rng(seed_base + 99999)

        print(f"\n  ── 目标: {tgt_ds} ──")

        for seed_off in range(n_seeds):
            seed = seed_base + seed_off

            # ── 条件1：Full-40 ────────────────────────────────────────────
            res = build_ovr(tgt_df, feature_cols, tgt_labels, seed)
            if res[0] is not None:
                m = train_model(res[0], res[2], seed)
                rows.append(dict(
                    semantic_class=sem_cls, target=tgt_ds,
                    source="—", condition="Full-40",
                    env_same=True, seed=seed,
                    **evaluate(m, res[1], res[3]),
                ))

            # ── 条件2：SHAP-10（特征集迁移，target 重训练）────────────────
            for src_ds in available:
                if src_ds == tgt_ds or src_ds not in datasets:
                    continue
                src_topk = get_topk(all_sigs[src_ds], mapping[src_ds],
                                    feature_cols, TOP_K)
                if not src_topk:
                    continue
                valid_feats = [f for f in src_topk if f in tgt_df.columns]
                if len(valid_feats) < TOP_K:
                    continue

                env_same = DS_ENV.get(src_ds) == DS_ENV.get(tgt_ds)

                # SHAP-10：target 数据训练，target 数据测试
                res2 = build_ovr(tgt_df, src_topk, tgt_labels, seed)
                if res2[0] is not None:
                    m2 = train_model(res2[0], res2[2], seed)
                    rows.append(dict(
                        semantic_class=sem_cls, target=tgt_ds,
                        source=src_ds, condition="SHAP-10",
                        env_same=env_same, seed=seed,
                        **evaluate(m2, res2[1], res2[3]),
                    ))

                # ── 条件3：SHAP-10-Direct（模型迁移，source 训练→target 测试）
                src_df = datasets[src_ds]
                # 在 source 上训练
                res_src = build_ovr(src_df, src_topk,
                                    mapping[src_ds], seed,
                                    test_size=TEST_SIZE)
                if res_src[0] is None:
                    continue
                m_direct = train_model(res_src[0], res_src[2], seed)

                # 在 target 上构建测试集（只用测试部分，不看训练数据）
                # 注意：这里需要独立的测试集，不能用 res2 的测试集
                # （res2 的测试集可能包含不同的样本分布）
                # 策略：用 target 的全部数据作为测试（模拟真实部署场景）
                pos_idx_tgt = tgt_df.index[
                    tgt_df["Attack"].isin(tgt_labels)
                ].to_numpy()
                neg_idx_tgt = tgt_df.index[
                    ~tgt_df["Attack"].isin(tgt_labels)
                ].to_numpy()

                if len(pos_idx_tgt) == 0:
                    continue

                # 为保持与其他条件可比，也取同等大小的测试集
                rng_tgt = np.random.default_rng(seed + 200000)
                n_pos_test = max(100, int(min(len(pos_idx_tgt), MAX_TRAIN_POS)
                                         * TEST_SIZE))
                n_neg_test = min(len(neg_idx_tgt),
                                 n_pos_test * NEG_RATIO_CAP)
                pos_test = rng_tgt.choice(pos_idx_tgt,
                                          n_pos_test, replace=False)
                neg_test = rng_tgt.choice(neg_idx_tgt,
                                          n_neg_test, replace=False)

                X_tgt_test = clean_X(np.vstack([
                    tgt_df.loc[pos_test, src_topk].to_numpy(),
                    tgt_df.loc[neg_test, src_topk].to_numpy(),
                ]))
                y_tgt_test = np.concatenate([
                    np.ones(len(pos_test),  dtype=np.int32),
                    np.zeros(len(neg_test), dtype=np.int32),
                ])

                rows.append(dict(
                    semantic_class=sem_cls, target=tgt_ds,
                    source=src_ds, condition="SHAP-10-Direct",
                    env_same=env_same, seed=seed,
                    **evaluate(m_direct, X_tgt_test, y_tgt_test),
                ))

            # ── 条件4：Random-10（下界基线）──────────────────────────────
            rand_feats = list(rng_rand.choice(
                feature_cols, size=TOP_K, replace=False))
            res4 = build_ovr(tgt_df, rand_feats, tgt_labels, seed)
            if res4[0] is not None:
                m4 = train_model(res4[0], res4[2], seed)
                rows.append(dict(
                    semantic_class=sem_cls, target=tgt_ds,
                    source="Random", condition="Random-10",
                    env_same=True, seed=seed,
                    **evaluate(m4, res4[1], res4[3]),
                ))

        print(f"    完成 {n_seeds} 轮")

    return rows


# ══════════════════════════════════════════════════════════════════════════════
# 统计分析
# ══════════════════════════════════════════════════════════════════════════════
def compute_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    三种对比的 Wilcoxon + Holm-Bonferroni：
      A: SHAP-10-Direct vs Random-10  （zero-shot 是否优于无信息基线）
      B: SHAP-10-Direct vs SHAP-10    （模型迁移 vs 特征集迁移的差距）
      C: SHAP-10        vs Random-10  （与分析3一致，作为参照）
    """
    comparisons = [
        ("SHAP-10-Direct", "Random-10",  "Direct_vs_Random"),
        ("SHAP-10-Direct", "SHAP-10",    "Direct_vs_SHAP10"),
        ("SHAP-10",        "Random-10",  "SHAP10_vs_Random"),
    ]

    stat_rows = []
    all_pvals = []  # 收集所有 p 值用于校正

    for cond_a, cond_b, comp_name in comparisons:
        sub_a = df[df["condition"] == cond_a]
        sub_b = df[df["condition"] == cond_b]

        for sem_cls in df["semantic_class"].unique():
            for tgt in df["target"].unique():
                # 对 SHAP-10 和 SHAP-10-Direct，还需要按 source 配对
                if cond_a in ("SHAP-10-Direct", "SHAP-10") and \
                   cond_b in ("SHAP-10-Direct", "SHAP-10"):
                    # 两者都有 source，需要按 source 配对
                    sources = df[
                        (df["condition"] == cond_a) &
                        (df["semantic_class"] == sem_cls) &
                        (df["target"] == tgt)
                    ]["source"].unique()
                    for src in sources:
                        a_vals = sub_a[
                            (sub_a["semantic_class"] == sem_cls) &
                            (sub_a["target"] == tgt) &
                            (sub_a["source"] == src)
                        ].sort_values("seed")["f1"].values

                        b_vals = sub_b[
                            (sub_b["semantic_class"] == sem_cls) &
                            (sub_b["target"] == tgt) &
                            (sub_b["source"] == src)
                        ].sort_values("seed")["f1"].values

                        n = min(len(a_vals), len(b_vals))
                        if n < 4:
                            continue
                        a_vals, b_vals = a_vals[:n], b_vals[:n]
                        wstat, wpval = wilcox_test(a_vals, b_vals)
                        d = cohen_d(a_vals, b_vals)
                        ci_lo, ci_hi = bootstrap_ci(a_vals - b_vals)
                        env_same = df[
                            (df["condition"] == cond_a) &
                            (df["target"] == tgt) &
                            (df["source"] == src)
                        ]["env_same"].iloc[0] if len(df[
                            (df["condition"] == cond_a) &
                            (df["target"] == tgt) &
                            (df["source"] == src)
                        ]) > 0 else None
                        stat_rows.append(dict(
                            comparison=comp_name,
                            semantic_class=sem_cls,
                            source=src, target=tgt,
                            pair=f"{src}\u2192{tgt}",
                            env_same=env_same,
                            n=n,
                            mean_a=round(float(a_vals.mean()), 4),
                            mean_b=round(float(b_vals.mean()), 4),
                            delta=round(float(a_vals.mean()-b_vals.mean()), 4),
                            cohen_d=round(d, 4),
                            ci95_delta_lo=round(ci_lo, 4),
                            ci95_delta_hi=round(ci_hi, 4),
                            wilcoxon_p=round(wpval, 6),
                        ))
                        all_pvals.append(wpval)
                else:
                    # cond_b 是 Random-10（无 source 维度），只按 target 配对
                    a_sub = sub_a[
                        (sub_a["semantic_class"] == sem_cls) &
                        (sub_a["target"] == tgt)
                    ]
                    b_sub = sub_b[
                        (sub_b["semantic_class"] == sem_cls) &
                        (sub_b["target"] == tgt)
                    ]
                    sources = a_sub["source"].unique()
                    for src in sources:
                        if src in ("—", "Random"):
                            continue
                        a_vals = a_sub[a_sub["source"] == src
                                       ].sort_values("seed")["f1"].values
                        b_vals = b_sub.sort_values("seed")["f1"].values
                        n = min(len(a_vals), len(b_vals))
                        if n < 4:
                            continue
                        a_vals, b_vals = a_vals[:n], b_vals[:n]
                        wstat, wpval = wilcox_test(a_vals, b_vals)
                        d = cohen_d(a_vals, b_vals)
                        ci_lo, ci_hi = bootstrap_ci(a_vals - b_vals)
                        env_same_vals = a_sub[
                            a_sub["source"] == src]["env_same"].values
                        env_same = bool(env_same_vals[0]) \
                            if len(env_same_vals) > 0 else None
                        stat_rows.append(dict(
                            comparison=comp_name,
                            semantic_class=sem_cls,
                            source=src, target=tgt,
                            pair=f"{src}\u2192{tgt}",
                            env_same=env_same,
                            n=n,
                            mean_a=round(float(a_vals.mean()), 4),
                            mean_b=round(float(b_vals.mean()), 4),
                            delta=round(float(a_vals.mean()-b_vals.mean()), 4),
                            cohen_d=round(d, 4),
                            ci95_delta_lo=round(ci_lo, 4),
                            ci95_delta_hi=round(ci_hi, 4),
                            wilcoxon_p=round(wpval, 6),
                        ))
                        all_pvals.append(wpval)

    stat_df = pd.DataFrame(stat_rows)

    # Holm-Bonferroni 校正（跨所有比较）
    if len(all_pvals) > 0:
        corrected = holm_bonferroni(all_pvals)
        stat_df["wilcoxon_p_corrected"] = [round(p, 6) for p in corrected]
        stat_df["significant"] = stat_df["wilcoxon_p_corrected"] < ALPHA

    return stat_df


# ══════════════════════════════════════════════════════════════════════════════
# 可视化
# ══════════════════════════════════════════════════════════════════════════════
def plot_results(raw_df: pd.DataFrame, stat_df: pd.DataFrame,
                 out_dir: Path, n_seeds: int):
    """
    主图：四条件箱线图（每个目标数据集一组），突出 SHAP-10-Direct
    上方子图：F1 分布
    下方子图：SHAP-10-Direct 与 SHAP-10 的差值（直接量化模型迁移损失）
    """
    sem_classes = raw_df["semantic_class"].unique()
    n_cls = len(sem_classes)

    fig, axes = plt.subplots(
        2, n_cls,
        figsize=(STYLE["full_width"] + 0.4, 5.0),
        gridspec_kw={"height_ratios": [2.5, 1.2]},
    )
    if n_cls == 1:
        axes = axes.reshape(2, 1)

    for ci, sem_cls in enumerate(sem_classes):
        ax_top = axes[0, ci]
        ax_bot = axes[1, ci]

        sub = raw_df[raw_df["semantic_class"] == sem_cls]
        tgt_list = [ds for ds in DS_ORDER
                    if ds in sub["target"].values]
        n_tgt = len(tgt_list)

        x_pos  = np.arange(n_tgt)
        bwidth = 0.17
        # 四条件偏移
        offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * bwidth

        # ── 上图：F1 箱线图 ─────────────────────────────────────────────
        for cond_i, cond in enumerate(COND_ORDER):
            cond_sub = sub[sub["condition"] == cond]
            color    = COND_COLORS[cond]

            for ti, tgt in enumerate(tgt_list):
                tgt_sub = cond_sub[cond_sub["target"] == tgt]

                if cond in ("Full-40", "Random-10"):
                    vals = tgt_sub["f1"].values
                else:
                    # SHAP-10 / SHAP-10-Direct：取所有 source 的均值
                    vals = tgt_sub.groupby("seed")["f1"].mean().values

                if len(vals) == 0:
                    continue

                bp = ax_top.boxplot(
                    vals,
                    positions=[x_pos[ti] + offsets[cond_i]],
                    widths=bwidth * 0.82,
                    patch_artist=True, vert=True,
                    boxprops=dict(facecolor=color, alpha=0.72),
                    medianprops=dict(color="white", linewidth=1.5),
                    whiskerprops=dict(color=color, linewidth=0.8),
                    capprops=dict(color=color, linewidth=0.8),
                    flierprops=dict(marker=".", markersize=2,
                                    color=color, alpha=0.4),
                )

        ax_top.set_xticks(x_pos)
        ax_top.set_xticklabels(tgt_list, fontsize=7)
        ax_top.set_ylabel("F1 score", fontsize=7.5)
        ax_top.set_ylim(
            max(0, raw_df["f1"].min() - 0.05),
            min(1.05, raw_df["f1"].max() + 0.03)
        )
        ax_top.axhline(0.9, color="#cccccc", lw=0.6, linestyle="--")
        ax_top.grid(axis="y", linewidth=0.22, alpha=0.35)
        ax_top.spines["top"].set_visible(False)
        ax_top.spines["right"].set_visible(False)
        panel = chr(ord("a") + ci)
        ax_top.set_title(
            f"({panel}) {sem_cls}  —  F1 across {n_seeds} seeds",
            loc="left", fontsize=8.5, fontweight="bold"
        )

        # ── 下图：Direct - SHAP10 差值（模型迁移代价）─────────────────
        direct_sub = sub[sub["condition"] == "SHAP-10-Direct"]
        shap10_sub = sub[sub["condition"] == "SHAP-10"]

        for ti, tgt in enumerate(tgt_list):
            d_tgt = direct_sub[direct_sub["target"] == tgt]
            s_tgt = shap10_sub[shap10_sub["target"] == tgt]
            sources = d_tgt["source"].unique()

            diffs = []
            for src in sources:
                d_vals = d_tgt[d_tgt["source"] == src
                               ].sort_values("seed")["f1"].values
                s_vals = s_tgt[s_tgt["source"] == src
                               ].sort_values("seed")["f1"].values
                n = min(len(d_vals), len(s_vals))
                if n > 0:
                    diffs.extend((d_vals[:n] - s_vals[:n]).tolist())

            if not diffs:
                continue

            diffs = np.array(diffs)
            med  = np.median(diffs)
            q25, q75 = np.percentile(diffs, [25, 75])
            color = "#d6604d" if med < 0 else "#4dac26"

            ax_bot.bar(x_pos[ti], med, width=0.55,
                       color=color, alpha=0.72, zorder=2)
            ax_bot.errorbar(x_pos[ti], med,
                            yerr=[[med - q25], [q75 - med]],
                            fmt="none", color=color,
                            elinewidth=1.0, capsize=3, capthick=0.9,
                            zorder=3)

        ax_bot.axhline(0, color="#555555", lw=0.8, linestyle="-")
        ax_bot.set_xticks(x_pos)
        ax_bot.set_xticklabels(tgt_list, fontsize=7)
        ax_bot.set_ylabel("\u0394 F1\n(Direct\u2212SHAP-10)", fontsize=6.5)
        ax_bot.grid(axis="y", linewidth=0.22, alpha=0.35)
        ax_bot.spines["top"].set_visible(False)
        ax_bot.spines["right"].set_visible(False)

    # ── 共享图例 ─────────────────────────────────────────────────────────
    handles = [
        mpatches.Patch(color=COND_COLORS[c], alpha=0.72, label=c)
        for c in COND_ORDER
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.04),
               fontsize=7, frameon=False, columnspacing=1.2)

    fig.suptitle(
        "Figure 11. SHAP-10-Direct (zero-shot model transfer) vs. "
        "feature-set transfer and baselines\n"
        f"Upper: F1 distribution ({n_seeds} seeds). "
        "Lower: F1 gap between SHAP-10-Direct and SHAP-10 "
        "(negative = model transfer penalty).",
        fontsize=9, y=1.01,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    save_fig(fig, out_dir / "analysis5_figure")
    print(f"  主图: analysis5_figure.pdf/.png")


# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════
def main():
    args    = parse_args()
    rq1_dir  = Path(args.rq1_dir)
    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_style()

    print(f"\n{'#'*65}")
    print(f"#  分析5：SHAP-10-Direct 零样本模型迁移实验（问题1修复）")
    print(f"#  n_seeds = {args.n_seeds}  类别 = {args.classes}")
    print(f"{'#'*65}")

    print("加载 RQ1 SHAP 签名...")
    all_sigs = load_shap_sigs(rq1_dir)
    print(f"已加载 {len(all_sigs)} 个数据集\n")

    all_rows = []
    for sem_cls in args.classes:
        if sem_cls not in SEMANTIC_MAPPING:
            continue
        rows = run_experiment(
            sem_cls, SEMANTIC_MAPPING[sem_cls],
            all_sigs, data_dir, args.n_seeds
        )
        all_rows.extend(rows)

    if not all_rows:
        print("无结果，退出")
        return

    raw_df = pd.DataFrame(all_rows)
    raw_csv = out_dir / "analysis5_raw_results.csv"
    raw_df.to_csv(raw_csv, index=False)
    print(f"\n原始数据: {raw_csv}  ({len(raw_df):,d} 行)")

    # ── 统计检验 ─────────────────────────────────────────────────────────
    print("\n计算统计检验（三种对比 × Holm-Bonferroni 校正）...")
    stat_df = compute_stats(raw_df)
    stat_csv = out_dir / "analysis5_statistical_tests.csv"
    stat_df.to_csv(stat_csv, index=False)
    print(f"统计检验: {stat_csv}")

    # ── 论文汇总表 ───────────────────────────────────────────────────────
    summary_df = stat_df[[
        "comparison", "semantic_class", "pair", "env_same",
        "mean_a", "mean_b", "delta",
        "cohen_d", "ci95_delta_lo", "ci95_delta_hi",
        "wilcoxon_p", "wilcoxon_p_corrected", "significant",
    ]]
    summary_csv = out_dir / "analysis5_summary_table.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"汇总表: {summary_csv}")

    # ── 可视化 ───────────────────────────────────────────────────────────
    print("\n绘图...")
    plot_results(raw_df, stat_df, out_dir, args.n_seeds)

    # ── 控制台结果 ───────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  结果汇总（Direct_vs_Random：零样本是否优于无信息基线）")
    print(f"{'='*65}")
    dr = stat_df[stat_df["comparison"] == "Direct_vs_Random"]
    n_sig = dr["significant"].sum() if "significant" in dr.columns else 0
    print(f"  显著配对（校正后）: {n_sig}/{len(dr)}")

    print(f"\n  结果汇总（Direct_vs_SHAP10：模型迁移 vs 特征集迁移差距）")
    print(f"{'='*65}")
    ds10 = stat_df[stat_df["comparison"] == "Direct_vs_SHAP10"]
    print(f"  {'配对':>20}  {'类别':>15}  {'Δ均值':>8}  "
          f"{'Cohen d':>8}  {'p(corr)':>10}  {'显著':>5}")
    print(f"  {'─'*20}  {'─'*15}  {'─'*8}  "
          f"{'─'*8}  {'─'*10}  {'─'*5}")
    for _, r in ds10.sort_values("delta").iterrows():
        sig = "*" if r.get("significant", False) else ""
        pcorr = r.get("wilcoxon_p_corrected", float("nan"))
        print(f"  {r['pair']:>20}  {r['semantic_class']:>15}  "
              f"{r['delta']:>+8.4f}  {r['cohen_d']:>8.4f}  "
              f"{pcorr:>10.4f}  {sig:>5}")

    print(f"\n  输出目录: {out_dir.resolve()}\n")

    # ── 论文文字提示 ─────────────────────────────────────────────────────
    print("="*65)
    print("  论文新增内容位置提示：")
    print("  1. Results 4.3 新增 4.3.4 节，描述 SHAP-10-Direct 结果")
    print("  2. Methodology 3.4 新增条件4描述")
    print("  3. Fig. 11 插入 4.3.4 节末尾")
    print("  4. Discussion 5.3 补充零样本迁移发现")
    print("="*65)


if __name__ == "__main__":
    main()
