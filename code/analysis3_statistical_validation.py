#!/usr/bin/env python3
"""
分析3：多种子重复实验 + 严格统计检验
======================================================
动机：
  原版：SHAP-10 单次 F1 vs Random-10×10 F1 分布 → Mann-Whitney U
        n_SHAP=1 vs n_Random=10，不构成有效推断比较

改进方案：
  对每个 source-target-class 配对，所有条件均做 N_SEEDS 次重复
  （不同随机划分种子），得到每个条件的 F1 分布，然后：
    1. Wilcoxon signed-rank test（配对，SHAP-10 vs Random-10）
    2. Mann-Whitney U（作为对比，与原结论一致性检验）
    3. Holm-Bonferroni 多重比较校正（跨所有配对）
    4. Cohen's d 效应量
    5. Bootstrap 95% CI

实验条件（与分析2一致，新增严格统计框架）：
  Full-40     : 全特征，每次重新划分train/test
  Target-10   : 目标数据集自身 Top-10 SHAP 特征
  SHAP-10     : 源数据集 Top-10 SHAP 特征 → 目标数据集
  Random-10   : 随机10特征（每次随机抽取，固定seed内唯一）

用法：
  # 先跑 DoS 验证（约20分钟）
  python analysis3_statistical_validation.py \\
      --rq1-dir ./results/rq1/ \\
      --data-dir ./processed/ \\
      --out ./results/analysis3/ \\
      --classes DoS --n-seeds 10

  # 全量跑（约60-90分钟）
  python analysis3_statistical_validation.py \\
      --rq1-dir ./results/rq1/ \\
      --data-dir ./processed/ \\
      --out ./results/analysis3/

输出：
  analysis3_raw_results.csv          — 全部重复实验原始F1
  analysis3_statistical_tests.csv    — 统计检验结果（含校正后p值）
  analysis3_summary_table.csv        — 论文Table VI替代表格（含CI）
  analysis3_figure.pdf/png           — 箱线图（含显著性标注）
"""

import argparse
import pickle
import warnings
from itertools import permutations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import wilcoxon, mannwhitneyu
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
N_SEEDS_DEFAULT = 10     # 每个条件重复次数（论文建议≥10）
BOOTSTRAP_N    = 1000    # Bootstrap CI 重复次数
ALPHA          = 0.05    # 显著性水平

STYLE = {"full_width": 7.16, "fig_dpi": 300, "font_size": 8}


def parse_args():
    p = argparse.ArgumentParser(description="分析3：多种子统计验证")
    p.add_argument("--rq1-dir",  required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out",      default="./results/analysis3/")
    p.add_argument("--classes",  nargs="+",
                   default=list(SEMANTIC_MAPPING.keys()))
    p.add_argument("--n-seeds",  type=int, default=N_SEEDS_DEFAULT,
                   help=f"每个条件的重复次数（默认{N_SEEDS_DEFAULT}，"
                        "论文建议≥10，稳健建议30）")
    return p.parse_args()


def setup_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size":   STYLE["font_size"],
        "axes.titlesize": 9, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "savefig.dpi": STYLE["fig_dpi"],
        "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
    })


def save_fig(fig, stem: Path):
    fig.savefig(str(stem) + ".pdf", format="pdf")
    fig.savefig(str(stem) + ".png", format="png")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# 数据和签名加载
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
    merged = weighted / total_w
    idx = np.argsort(merged)[::-1][:k]
    return [feature_cols[i] for i in idx]


def load_dataset(data_dir: Path, ds: str) -> pd.DataFrame | None:
    p = data_dir / DS_CSV[ds]
    return pd.read_csv(p, low_memory=False) if p.exists() else None


# ══════════════════════════════════════════════════════════════════════════════
# 数据集构建（带随机种子）
# ══════════════════════════════════════════════════════════════════════════════
def build_dataset(df: pd.DataFrame, feature_cols: list,
                  target_labels: list, seed: int):
    """
    构建 OvR 数据集。
    seed 控制正负例采样和 train/test 划分，实现多种子重复。
    """
    attack_col = "Attack"
    use_feats  = [c for c in feature_cols
                  if c in df.columns and c not in ("Label", "Attack")]

    pos_mask = df[attack_col].isin(target_labels)
    pos_idx  = df.index[pos_mask].to_numpy()
    neg_idx  = df.index[~pos_mask].to_numpy()

    if len(pos_idx) == 0:
        return None, None, None, None

    rng = np.random.default_rng(seed)

    if len(pos_idx) > MAX_TRAIN_POS:
        pos_idx = rng.choice(pos_idx, MAX_TRAIN_POS, replace=False)
    neg_cap = len(pos_idx) * NEG_RATIO_CAP
    if len(neg_idx) > neg_cap:
        neg_idx = rng.choice(neg_idx, int(neg_cap), replace=False)

    X_pos = df.loc[pos_idx, use_feats].to_numpy(dtype=np.float64)
    X_neg = df.loc[neg_idx, use_feats].to_numpy(dtype=np.float64)
    X = np.vstack([X_pos, X_neg])
    y = np.concatenate([
        np.ones(len(pos_idx), dtype=np.int32),
        np.zeros(len(neg_idx), dtype=np.int32),
    ])

    # 清理极值
    X[~np.isfinite(X)] = 0.0
    large = np.abs(X) > 1e15
    if large.any():
        for ci in np.where(large.any(axis=0))[0]:
            col = X[:, ci]
            valid = col[~large[:, ci]]
            col[large[:, ci]] = float(np.median(valid)) if len(valid) > 0 else 0.0

    # 使用 seed 控制 train/test 划分
    return train_test_split(X, y, test_size=TEST_SIZE,
                            random_state=seed, stratify=y)


def run_xgb(X_train, X_test, y_train, y_test, seed: int) -> dict:
    y_train = np.asarray(y_train, dtype=np.int32)
    y_test  = np.asarray(y_test,  dtype=np.int32)
    params  = XGB_PARAMS.copy()
    params["scale_pos_weight"] = int((y_train == 0).sum()) / max(int((y_train == 1).sum()), 1)
    params["random_state"] = seed

    m = xgb.XGBClassifier(**params)
    m.fit(X_train, y_train, verbose=False)
    y_pred  = m.predict(X_test)
    y_proba = m.predict_proba(X_test)[:, 1]
    return {
        "f1":      float(f1_score(y_test, y_pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_test, y_proba)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 统计检验工具
# ══════════════════════════════════════════════════════════════════════════════
def bootstrap_ci(values: np.ndarray, n: int = BOOTSTRAP_N,
                 alpha: float = ALPHA, seed: int = 0) -> tuple[float, float]:
    """Bootstrap 置信区间"""
    rng  = np.random.default_rng(seed)
    boot = np.array([rng.choice(values, len(values), replace=True).mean()
                     for _ in range(n)])
    lo = float(np.percentile(boot, 100 * alpha / 2))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return lo, hi


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d 效应量"""
    na, nb = len(a), len(b)
    pooled_std = np.sqrt(
        ((na - 1) * a.std(ddof=1)**2 + (nb - 1) * b.std(ddof=1)**2)
        / (na + nb - 2)
    )
    return float((a.mean() - b.mean()) / pooled_std) if pooled_std > 0 else 0.0


def holm_bonferroni(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni 校正，返回校正后的 p 值"""
    n = len(pvals)
    indexed = sorted(enumerate(pvals), key=lambda x: x[1])
    corrected = [None] * n
    running_min = 1.0
    for rank, (orig_idx, p) in enumerate(reversed(indexed)):
        k = n - rank  # 从最大 p 开始
        corrected_p = min(running_min, p * k)
        running_min = corrected_p
        corrected[orig_idx] = corrected_p
    return corrected


# ══════════════════════════════════════════════════════════════════════════════
# 主实验：多种子重复
# ══════════════════════════════════════════════════════════════════════════════
def run_repeated_experiment(sem_cls: str, mapping: dict,
                            all_sigs: dict, data_dir: Path,
                            n_seeds: int) -> list[dict]:
    """
    对一个语义类别的所有 source-target 配对，
    每个条件重复 n_seeds 次（不同随机种子控制数据划分）。
    返回原始实验记录列表。
    """
    print(f"\n  语义类别: {sem_cls}  (每条件重复 {n_seeds} 次)")
    print(f"  {'─'*60}")

    available = [ds for ds in DS_ORDER
                 if ds in mapping and ds in all_sigs]

    # 获取特征列
    feature_cols = None
    for ds in available:
        labels = mapping[ds]
        matched = {l: all_sigs[ds][l] for l in labels if l in all_sigs[ds]}
        if matched:
            feature_cols = list(matched.values())[0]["feature_cols"]
            break
    if feature_cols is None:
        return []

    # 加载数据集
    print("  加载数据集...")
    datasets = {}
    for ds in available:
        df = load_dataset(data_dir, ds)
        if df is not None:
            datasets[ds] = df
            print(f"    {ds}: {len(df):,d} 行")

    rows = []
    seed_base = 42  # 基础种子，各轮加偏移

    for tgt_ds in available:
        if tgt_ds not in datasets:
            continue
        tgt_df     = datasets[tgt_ds]
        tgt_labels = mapping[tgt_ds]

        print(f"\n  ── 目标: {tgt_ds} ──")

        # ── Target-10 特征（目标数据集自身 Top-10）────────────────────────
        tgt_topk = get_topk(all_sigs[tgt_ds], tgt_labels, feature_cols, TOP_K)

        # ── 为每个条件收集 n_seeds 次 F1 ─────────────────────────────────
        for seed_offset in range(n_seeds):
            seed = seed_base + seed_offset

            # 条件1：Full-40
            res = build_dataset(tgt_df, feature_cols, tgt_labels, seed)
            if res[0] is not None:
                m = run_xgb(*res, seed)
                rows.append(dict(
                    semantic_class=sem_cls, target=tgt_ds,
                    source="—", condition="Full-40",
                    env_same=True, seed=seed,
                    f1=round(m["f1"], 6), auc_roc=round(m["auc_roc"], 6),
                ))

            # 条件2：Target-10
            if tgt_topk:
                res = build_dataset(tgt_df, tgt_topk, tgt_labels, seed)
                if res[0] is not None:
                    m = run_xgb(*res, seed)
                    rows.append(dict(
                        semantic_class=sem_cls, target=tgt_ds,
                        source=tgt_ds, condition="Target-10",
                        env_same=True, seed=seed,
                        f1=round(m["f1"], 6), auc_roc=round(m["auc_roc"], 6),
                    ))

            # 条件3：SHAP-10（来自各个源数据集）
            for src_ds in available:
                if src_ds == tgt_ds or src_ds not in datasets:
                    continue
                src_topk = get_topk(all_sigs[src_ds], mapping[src_ds],
                                    feature_cols, TOP_K)
                if not src_topk:
                    continue
                valid = [f for f in src_topk if f in tgt_df.columns]
                if len(valid) < TOP_K:
                    continue
                res = build_dataset(tgt_df, src_topk, tgt_labels, seed)
                if res[0] is not None:
                    m = run_xgb(*res, seed)
                    env_same = DS_ENV.get(src_ds) == DS_ENV.get(tgt_ds)
                    rows.append(dict(
                        semantic_class=sem_cls, target=tgt_ds,
                        source=src_ds, condition="SHAP-10",
                        env_same=env_same, seed=seed,
                        f1=round(m["f1"], 6), auc_roc=round(m["auc_roc"], 6),
                    ))

            # 条件4：Random-10（每个 seed 抽一组不同的随机特征）
            rng = np.random.default_rng(seed + 10000)  # 避免与其他条件重叠
            rand_feats = list(rng.choice(feature_cols, size=TOP_K, replace=False))
            res = build_dataset(tgt_df, rand_feats, tgt_labels, seed)
            if res[0] is not None:
                m = run_xgb(*res, seed)
                rows.append(dict(
                    semantic_class=sem_cls, target=tgt_ds,
                    source="Random", condition="Random-10",
                    env_same=True, seed=seed,
                    f1=round(m["f1"], 6), auc_roc=round(m["auc_roc"], 6),
                ))

        # 进度打印
        n_done = seed_offset + 1
        print(f"    完成 {n_done}/{n_seeds} 轮")

    return rows


# ══════════════════════════════════════════════════════════════════════════════
# 统计分析
# ══════════════════════════════════════════════════════════════════════════════
def compute_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    对每个 (semantic_class, source→target) 配对计算：
      - Wilcoxon signed-rank test（配对：同一 seed 的 SHAP-10 vs Random-10）
      - Mann-Whitney U（非配对，用于与原版对比）
      - Cohen's d 效应量
      - Bootstrap 95% CI（SHAP-10 均值）
    """
    stat_rows = []

    shap_df   = df[df["condition"] == "SHAP-10"]
    random_df = df[df["condition"] == "Random-10"]

    for sem_cls in df["semantic_class"].unique():
        cls_shap   = shap_df[shap_df["semantic_class"] == sem_cls]
        cls_random = random_df[random_df["semantic_class"] == sem_cls]

        for tgt in cls_shap["target"].unique():
            tgt_random = cls_random[cls_random["target"] == tgt]["f1"].values

            for src in cls_shap[cls_shap["target"] == tgt]["source"].unique():
                mask = (cls_shap["target"] == tgt) & (cls_shap["source"] == src)
                shap_f1s = cls_shap[mask].sort_values("seed")["f1"].values

                # 对齐种子（配对检验的前提）
                rand_f1s = cls_random[
                    cls_random["target"] == tgt
                ].sort_values("seed")["f1"].values

                # 长度对齐（取最短）
                n = min(len(shap_f1s), len(rand_f1s))
                shap_f1s = shap_f1s[:n]
                rand_f1s = rand_f1s[:n]

                if n < 4:  # 样本太少，跳过
                    continue

                # Wilcoxon signed-rank（配对）
                diff = shap_f1s - rand_f1s
                if np.all(diff == 0):
                    wilcox_stat, wilcox_p = np.nan, 1.0
                else:
                    try:
                        res = wilcoxon(shap_f1s, rand_f1s,
                                       alternative="greater")
                        wilcox_stat, wilcox_p = float(res.statistic), float(res.pvalue)
                    except Exception:
                        wilcox_stat, wilcox_p = np.nan, 1.0

                # Mann-Whitney U（非配对，用于对比）
                try:
                    mw_stat, mw_p = mannwhitneyu(
                        shap_f1s, rand_f1s, alternative="greater"
                    )
                except Exception:
                    mw_stat, mw_p = np.nan, 1.0

                # Cohen's d
                d = cohen_d(shap_f1s, rand_f1s)

                # Bootstrap CI
                ci_lo, ci_hi = bootstrap_ci(shap_f1s)

                env_same = (DS_ENV.get(src, "?") == DS_ENV.get(tgt, "?"))

                stat_rows.append({
                    "semantic_class":   sem_cls,
                    "source":           src,
                    "target":           tgt,
                    "pair":             f"{src}→{tgt}",
                    "env_same":         env_same,
                    "env_src":          DS_ENV.get(src, "?"),
                    "env_tgt":          DS_ENV.get(tgt, "?"),
                    "n_seeds":          n,
                    "shap10_mean":      round(float(shap_f1s.mean()), 4),
                    "shap10_std":       round(float(shap_f1s.std(ddof=1)), 4),
                    "shap10_ci95_lo":   round(ci_lo, 4),
                    "shap10_ci95_hi":   round(ci_hi, 4),
                    "random10_mean":    round(float(rand_f1s.mean()), 4),
                    "random10_std":     round(float(rand_f1s.std(ddof=1)), 4),
                    "delta_mean":       round(float(shap_f1s.mean() - rand_f1s.mean()), 4),
                    "cohen_d":          round(d, 4),
                    "wilcoxon_stat":    round(wilcox_stat, 4) if not np.isnan(wilcox_stat) else np.nan,
                    "wilcoxon_p":       round(wilcox_p, 6),
                    "mw_p":             round(float(mw_p), 6),
                    "significant_wilcox": wilcox_p < ALPHA,
                })

    stat_df = pd.DataFrame(stat_rows)

    # Holm-Bonferroni 多重比较校正（对 Wilcoxon p 值）
    if len(stat_df) > 0:
        raw_p = stat_df["wilcoxon_p"].tolist()
        corrected = holm_bonferroni(raw_p)
        stat_df["wilcoxon_p_corrected"] = [round(p, 6) for p in corrected]
        stat_df["significant_corrected"] = stat_df["wilcoxon_p_corrected"] < ALPHA

    return stat_df


# ══════════════════════════════════════════════════════════════════════════════
# 可视化
# ══════════════════════════════════════════════════════════════════════════════
def plot_results(raw_df: pd.DataFrame, stat_df: pd.DataFrame,
                 out_dir: Path, n_seeds: int):
    """
    主结果图：每个 source→target 配对的四条件箱线图
    叠加统计显著性标注（校正后 p 值）
    """
    sem_classes = raw_df["semantic_class"].unique()
    n_cls = len(sem_classes)

    cond_colors = {
        "Full-40":   "#2166ac",
        "Target-10": "#4dac26",
        "SHAP-10":   "#d6604d",
        "Random-10": "#888888",
    }
    cond_order = ["Full-40", "Target-10", "SHAP-10", "Random-10"]

    # 每个语义类别一行子图，每行按目标数据集分列
    fig, axes = plt.subplots(1, n_cls,
                             figsize=(STYLE["full_width"], 3.2))
    if n_cls == 1:
        axes = [axes]

    for ax, sem_cls in zip(axes, sem_classes):
        sub = raw_df[raw_df["semantic_class"] == sem_cls]
        shap_sub = sub[sub["condition"] == "SHAP-10"]

        tgt_list = [ds for ds in DS_ORDER
                    if ds in sub["target"].values]
        n_tgt = len(tgt_list)
        if n_tgt == 0:
            ax.set_visible(False)
            continue

        x_pos   = np.arange(n_tgt)
        bwidth  = 0.18
        offsets = np.linspace(-1.5 * bwidth, 1.5 * bwidth, 4)

        for ci, cond in enumerate(cond_order):
            cond_sub = sub[sub["condition"] == cond]
            for ti, tgt in enumerate(tgt_list):
                vals = cond_sub[cond_sub["target"] == tgt]["f1"].values
                if len(vals) == 0:
                    continue

                bp = ax.boxplot(
                    vals,
                    positions=[x_pos[ti] + offsets[ci]],
                    widths=bwidth * 0.85,
                    patch_artist=True,
                    vert=True,
                    boxprops=dict(facecolor=cond_colors[cond], alpha=0.70),
                    medianprops=dict(color="white", linewidth=1.5),
                    whiskerprops=dict(color=cond_colors[cond], linewidth=0.8),
                    capprops=dict(color=cond_colors[cond], linewidth=0.8),
                    flierprops=dict(marker=".", markersize=2,
                                    color=cond_colors[cond], alpha=0.4),
                    showfliers=True,
                )

        # 显著性标注（SHAP-10 校正后 p）
        if stat_df is not None and len(stat_df) > 0:
            cls_stat = stat_df[stat_df["semantic_class"] == sem_cls]
            for ti, tgt in enumerate(tgt_list):
                tgt_stat = cls_stat[cls_stat["target"] == tgt]
                if tgt_stat.empty:
                    continue
                # 每个源数据集的最高 y 位置上方标注
                top_y = sub[sub["target"] == tgt]["f1"].max()
                for _, row in tgt_stat.iterrows():
                    p_corr = row.get("wilcoxon_p_corrected", 1.0)
                    if p_corr < 0.001:
                        sig = "***"
                    elif p_corr < 0.01:
                        sig = "**"
                    elif p_corr < 0.05:
                        sig = "*"
                    else:
                        sig = ""
                    if sig:
                        ax.text(x_pos[ti], top_y + 0.005, sig,
                                ha="center", va="bottom",
                                fontsize=7, color="#333333")
                        top_y += 0.015

        ax.set_xticks(x_pos)
        ax.set_xticklabels(tgt_list, fontsize=7)
        ax.set_ylabel("F1 score")
        ax.set_ylim(ax.get_ylim()[0], min(1.05, ax.get_ylim()[1] + 0.05))
        ax.axhline(0.9, color="#cccccc", linestyle="--", linewidth=0.6)
        ax.grid(axis="y", linewidth=0.25, alpha=0.4)

        panel = chr(ord("a") + list(sem_classes).index(sem_cls))
        ax.set_title(f"({panel}) {sem_cls}  (n={n_seeds} seeds per condition)",
                     loc="left", fontsize=8, fontweight="bold")

    # 图例
    handles = [
        mpatches.Patch(color=cond_colors[c], alpha=0.70, label=c)
        for c in cond_order
    ]
    handles += [
        plt.Line2D([0], [0], color="none", label="* p<0.05  ** p<0.01  *** p<0.001"),
        plt.Line2D([0], [0], color="none", label="(Wilcoxon signed-rank, Holm-Bonferroni corrected)"),
    ]
    fig.legend(handles=handles, loc="lower center",
               ncol=3, bbox_to_anchor=(0.5, -0.12),
               fontsize=7, frameon=False, columnspacing=1.0)

    fig.suptitle(
        f"Figure 7 (revised). SHAP-guided feature transfer: {n_seeds}-seed repeated experiment\n"
        f"Boxes show F1 distribution across {n_seeds} random train/test splits",
        fontsize=9, y=1.02
    )
    fig.tight_layout()
    save_fig(fig, out_dir / "analysis3_figure")
    print(f"  主图: analysis3_figure.pdf/.png")


def plot_ci_summary(stat_df: pd.DataFrame, out_dir: Path):
    """
    CI 汇总图：SHAP-10 vs Random-10 均值差 + 95% Bootstrap CI
    按是否显著（校正后）分色
    """
    if stat_df is None or len(stat_df) == 0:
        return

    fig, ax = plt.subplots(figsize=(STYLE["full_width"], 2.8))

    y_pos = 0
    yticks, ylabels = [], []

    for _, row in stat_df.sort_values(
        ["semantic_class", "delta_mean"], ascending=[True, False]
    ).iterrows():
        color = "#d6604d" if row.get("significant_corrected", False) else "#aaaaaa"
        delta  = row["delta_mean"]
        ci_lo  = row["shap10_ci95_lo"] - row["random10_mean"]
        ci_hi  = row["shap10_ci95_hi"] - row["random10_mean"]

        # xerr = [左误差, 右误差]，必须非负
        # CI 是 SHAP-10 均值的区间，转换为相对于 delta 的偏差
        shap_mean = row["shap10_mean"]
        err_lo = max(0.0, shap_mean - row["shap10_ci95_lo"])  # 左侧（向下）
        err_hi = max(0.0, row["shap10_ci95_hi"] - shap_mean)  # 右侧（向上）
        ax.errorbar(delta, y_pos,
                    xerr=[[err_lo], [err_hi]],
                    fmt="o", color=color, markersize=5,
                    elinewidth=1.2, capsize=3, capthick=1.0)
        ax.scatter(delta, y_pos, s=40, color=color, zorder=3)

        yticks.append(y_pos)
        ylabels.append(f"{row['pair']} ({row['semantic_class']})")
        y_pos += 1

    ax.axvline(0, color="#333333", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=6.5)
    ax.set_xlabel("SHAP-10 − Random-10 mean F1 (95% Bootstrap CI)")
    ax.set_title(
        "SHAP-10 vs Random-10 performance gain\n"
        "(red = significant after Holm-Bonferroni correction, grey = not significant)",
        fontsize=8, loc="left"
    )
    ax.grid(axis="x", linewidth=0.3, alpha=0.4)
    fig.tight_layout()
    save_fig(fig, out_dir / "analysis3_ci_summary")
    print(f"  CI 汇总图: analysis3_ci_summary.pdf/.png")


# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════
def main():
    args = parse_args()
    rq1_dir  = Path(args.rq1_dir)
    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_seeds  = args.n_seeds

    setup_style()

    print(f"\n{'#'*65}")
    print(f"#  分析3：多种子重复统计验证（问题2修复）")
    print(f"#  重复次数: {n_seeds} seeds/条件")
    print(f"#  分析类别: {args.classes}")
    print(f"{'#'*65}")
    print(f"\n预估耗时：{n_seeds} × 每轮约3-5分钟 ≈ {n_seeds*4//60}h{n_seeds*4%60}min\n")

    print("加载 RQ1 SHAP 签名...")
    all_sigs = load_shap_sigs(rq1_dir)
    print(f"已加载 {len(all_sigs)} 个数据集\n")

    all_rows = []
    for sem_cls in args.classes:
        if sem_cls not in SEMANTIC_MAPPING:
            continue
        rows = run_repeated_experiment(
            sem_cls, SEMANTIC_MAPPING[sem_cls],
            all_sigs, data_dir, n_seeds
        )
        all_rows.extend(rows)

    if not all_rows:
        print("无结果，退出")
        return

    raw_df = pd.DataFrame(all_rows)

    # ── 保存原始数据 ─────────────────────────────────────────────────────────
    raw_csv = out_dir / "analysis3_raw_results.csv"
    raw_df.to_csv(raw_csv, index=False)
    print(f"\n原始数据: {raw_csv}  ({len(raw_df):,d} 行)")

    # ── 统计检验 ─────────────────────────────────────────────────────────────
    print("\n计算统计检验...")
    stat_df = compute_statistics(raw_df)

    stat_csv = out_dir / "analysis3_statistical_tests.csv"
    stat_df.to_csv(stat_csv, index=False)
    print(f"统计检验结果: {stat_csv}")

    # ── 论文用汇总表 ─────────────────────────────────────────────────────────
    summary_cols = [
        "semantic_class", "pair", "env_same",
        "n_seeds",
        "shap10_mean", "shap10_std", "shap10_ci95_lo", "shap10_ci95_hi",
        "random10_mean", "random10_std",
        "delta_mean", "cohen_d",
        "wilcoxon_p", "wilcoxon_p_corrected",
        "significant_wilcox", "significant_corrected",
    ]
    summary_df = stat_df[[c for c in summary_cols if c in stat_df.columns]]
    summary_csv = out_dir / "analysis3_summary_table.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"论文汇总表: {summary_csv}")

    # ── 可视化 ───────────────────────────────────────────────────────────────
    print("\n绘图...")
    plot_results(raw_df, stat_df, out_dir, n_seeds)
    plot_ci_summary(stat_df, out_dir)

    # ── 控制台结果汇总 ───────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  统计检验结果汇总（Wilcoxon signed-rank, Holm-Bonferroni 校正）")
    print(f"{'='*65}")
    print(f"\n  {'配对':>20}  {'类别':>15}  {'Δ均值':>8}  "
          f"{'Cohen d':>8}  {'p(raw)':>10}  {'p(corr)':>10}  {'显著*':>5}")
    print(f"  {'─'*20}  {'─'*15}  {'─'*8}  "
          f"{'─'*8}  {'─'*10}  {'─'*10}  {'─'*5}")

    for _, r in stat_df.sort_values(
        ["semantic_class", "delta_mean"], ascending=[True, False]
    ).iterrows():
        sig_raw  = "*" if r.get("significant_wilcox",  False) else ""
        sig_corr = "*" if r.get("significant_corrected", False) else ""
        print(f"  {r['pair']:>20}  {r['semantic_class']:>15}  "
              f"{r['delta_mean']:>+8.4f}  {r['cohen_d']:>8.4f}  "
              f"{r['wilcoxon_p']:>10.4f}  "
              f"{r.get('wilcoxon_p_corrected', float('nan')):>10.4f}  "
              f"{'*(raw)' if sig_raw else '':>5}")

    n_sig = stat_df.get("significant_corrected", pd.Series(dtype=bool)).sum()
    n_tot = len(stat_df)
    print(f"\n  校正后显著配对: {n_sig}/{n_tot} "
          f"({100*n_sig/n_tot:.0f}%)\n")
    print(f"  输出目录: {out_dir.resolve()}\n")


if __name__ == "__main__":
    main()
