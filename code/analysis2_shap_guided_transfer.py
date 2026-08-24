#!/usr/bin/env python3
"""
分析2：SHAP-guided 特征迁移实验（RQ3）
补充分析 — 将负面结论转化为建设性贡献

研究问题：用 Dataset A 学到的 SHAP 特征签名指导特征选择，
          在 Dataset B 上训练的模型性能会如何变化？
          迁移效果是否与网络环境相似度相关？

实验设计：
  对 DoS（主实验，4数据集）、Reconnaissance（3数据集）各做：

  对每个"源→目标"数据集对：
    条件1 [Full-40]:   用全部40特征在目标数据集上训练 XGBoost，作为上界基线
    条件2 [SHAP-10]:   用源数据集 Top-10 SHAP 特征在目标数据集上训练
    条件3 [Random-10]: 随机10特征重复10次取均值，作为下界基线
    条件4 [Target-10]: 用目标数据集自己的 Top-10 SHAP 特征训练，作为理论上界

  核心指标：F1（正类）和 AUC-ROC
  统计检验：SHAP-10 vs Random-10 的 Mann-Whitney U 检验

输入：
  - results/rq1/*_rq1_shap_values.pkl
  - processed/*_processed.csv

输出（保存到 --out 目录）：
  analysis2_transfer_results.csv    — 全量实验结果
  analysis2_transfer_summary.csv    — 按语义类别和数据集对汇总
  analysis2_transfer_figure.pdf/png — 主结果图（Fig 7）
  analysis2_env_correlation.csv     — 迁移性 vs 环境相似度相关分析

用法：
  python analysis2_shap_guided_transfer.py \\
      --rq1-dir ./results/rq1/ \\
      --data-dir ./processed/ \\
      --out ./results/analysis2/
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

# 各数据集的 CSV 文件名
DS_CSV = {v: f"NF-{k.split('NF-')[1]}_processed.csv"
          for k, v in DATASET_SHORT.items()}
DS_CSV = {
    "UNSW": "NF-UNSW-NB15-v2_processed.csv",
    "CIC":  "NF-CSE-CIC-IDS2018-v2_processed.csv",
    "ToN":  "NF-ToN-IoT-v2_processed.csv",
    "BoT":  "NF-BoT-IoT-v2_processed.csv",
}

# 网络环境标注（用于计算环境相似度）
DS_ENV = {
    "UNSW": "IT",
    "CIC":  "IT",
    "ToN":  "IoT",
    "BoT":  "IoT",
}

# 跨数据集攻击类别映射
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

# XGBoost 参数（与 RQ1 保持一致）
XGB_PARAMS = {
    "n_estimators":     300,
    "max_depth":        6,
    "learning_rate":    0.1,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "eval_metric":      "logloss",
    "random_state":     42,
    "n_jobs":           -1,
    "tree_method":      "hist",
    "missing":          np.nan,
}

TOP_K          = 10    # SHAP 特征数
N_RANDOM_REPS  = 10   # 随机基线重复次数
TEST_SIZE      = 0.20
MAX_TRAIN_POS  = 30_000  # 正例上限（避免大类过慢）
NEG_RATIO_CAP  = 10      # 负例/正例比上限
RANDOM_SEED    = 42

STYLE = {"full_width": 7.16, "fig_dpi": 300, "font_size": 8}


def parse_args():
    p = argparse.ArgumentParser(description="分析2：SHAP-guided 特征迁移实验")
    p.add_argument("--rq1-dir",  required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out",      default="./results/analysis2/")
    p.add_argument("--classes",  nargs="+",
                   default=list(SEMANTIC_MAPPING.keys()),
                   help="只分析指定语义类别（默认全部）")
    return p.parse_args()


def setup_style():
    plt.rcParams.update({
        "font.family":   "DejaVu Sans",
        "font.size":     STYLE["font_size"],
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "savefig.dpi":   STYLE["fig_dpi"],
        "savefig.bbox":  "tight",
        "savefig.pad_inches": 0.05,
    })


def save_fig(fig, path_stem: Path):
    fig.savefig(str(path_stem) + ".pdf", format="pdf")
    fig.savefig(str(path_stem) + ".png", format="png")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# 数据加载
# ══════════════════════════════════════════════════════════════════════════════
def load_shap_sigs(rq1_dir: Path) -> dict:
    all_sigs = {}
    for ds_full, ds_short in DATASET_SHORT.items():
        pkl = rq1_dir / f"{ds_full}_rq1_shap_values.pkl"
        if pkl.exists():
            with open(pkl, "rb") as f:
                all_sigs[ds_short] = pickle.load(f)
    return all_sigs


def get_topk_features(sigs: dict, raw_labels: list,
                      feature_cols: list, k: int) -> list:
    """
    从多个原始标签（如 DoS 的4个变体）合并后取 Top-K 特征。
    合并方式：按 n_pos_shap 加权平均 abs_mean_shap。
    """
    matched = {lbl: sigs[lbl] for lbl in raw_labels if lbl in sigs}
    if not matched:
        return []

    total_w = 0
    weighted = np.zeros(len(feature_cols))
    for sig in matched.values():
        w = sig.get("n_pos_shap", 1)
        weighted += sig["abs_mean_shap"] * w
        total_w  += w
    merged = weighted / total_w
    top_idx = np.argsort(merged)[::-1][:k]
    return [feature_cols[i] for i in top_idx]


def load_dataset(data_dir: Path, ds_short: str) -> pd.DataFrame:
    csv_path = data_dir / DS_CSV[ds_short]
    if not csv_path.exists():
        print(f"  文件不存在: {csv_path}")
        return None
    df = pd.read_csv(csv_path, low_memory=False)
    return df


def build_binary_dataset(df: pd.DataFrame, feature_cols: list,
                         target_labels: list, seed: int):
    """
    构建二分类数据集：目标攻击类 vs 其余所有类。
    完全用 numpy 操作，避免 pandas DataFrame 的类型推断陷阱。
    """
    attack_col = "Attack"

    # 过滤特征列（排除标签列）
    exclude = {"Label", "Attack", "label", "attack", "_y"}
    use_feats = [c for c in feature_cols
                 if c in df.columns and c not in exclude]

    pos_mask = df[attack_col].isin(target_labels)
    neg_mask = ~pos_mask

    n_pos_total = int(pos_mask.sum())
    if n_pos_total == 0:
        return None, None, None, None

    # 正例/负例索引
    pos_idx = df.index[pos_mask].to_numpy()
    neg_idx = df.index[neg_mask].to_numpy()

    rng = np.random.default_rng(seed)

    # 正例上限
    if len(pos_idx) > MAX_TRAIN_POS:
        pos_idx = rng.choice(pos_idx, size=MAX_TRAIN_POS, replace=False)

    # 负例上限
    neg_cap = len(pos_idx) * NEG_RATIO_CAP
    if len(neg_idx) > neg_cap:
        neg_idx = rng.choice(neg_idx, size=int(neg_cap), replace=False)

    # 直接用 numpy 提取特征矩阵，完全绕开 pandas concat
    X_pos = df.loc[pos_idx, use_feats].to_numpy(dtype=np.float64)
    X_neg = df.loc[neg_idx, use_feats].to_numpy(dtype=np.float64)
    X = np.vstack([X_pos, X_neg])

    # 标签：纯 numpy int32 数组
    y = np.concatenate([
        np.ones(len(pos_idx), dtype=np.int32),
        np.zeros(len(neg_idx), dtype=np.int32),
    ])

    # 清理 inf 和极大值
    inf_mask = ~np.isfinite(X)
    if inf_mask.any():
        X[inf_mask] = 0.0
    large_mask = np.abs(X) > 1e15
    if large_mask.any():
        for ci in np.where(large_mask.any(axis=0))[0]:
            col = X[:, ci]
            valid = col[~large_mask[:, ci]]
            med = float(np.median(valid)) if len(valid) > 0 else 0.0
            col[large_mask[:, ci]] = med

    # 安全检查
    assert X.shape[0] == len(y), f"X/y 长度不匹配: {X.shape[0]} vs {len(y)}"
    assert X.shape[1] == len(use_feats), f"特征数不匹配: {X.shape[1]} vs {len(use_feats)}"
    assert set(np.unique(y)) == {0, 1}, f"y 值异常: {np.unique(y)}"

    return train_test_split(X, y, test_size=TEST_SIZE,
                            random_state=seed, stratify=y)


def run_xgb(X_train, X_test, y_train, y_test) -> dict:
    """训练 XGBoost 并返回评估指标"""
    # 强制确保 y 类型正确（防御性编程）
    y_train = np.asarray(y_train, dtype=np.int32)
    y_test  = np.asarray(y_test, dtype=np.int32)

    params = XGB_PARAMS.copy()
    params["objective"] = "binary:logistic"
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    params["scale_pos_weight"] = n_neg / max(n_pos, 1)

    model = xgb.XGBClassifier(**params)
    # 不传 eval_set，避免潜在的验证集标签编码冲突
    model.fit(X_train, y_train, verbose=False)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "f1":      round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "auc_roc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "n_pos_test": int((y_test == 1).sum()),
        "n_neg_test": int((y_test == 0).sum()),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 主实验
# ══════════════════════════════════════════════════════════════════════════════
def run_transfer_experiment(sem_cls: str, mapping: dict,
                            all_sigs: dict, data_dir: Path,
                            out_dir: Path) -> list:
    """
    对一个语义类别（如 DoS）做完整的特征迁移实验。
    返回结果行列表。
    """
    print(f"\n  语义类别: {sem_cls}")
    print(f"  {'─'*56}")

    # 确定可用数据集
    available = [ds for ds in DS_ORDER
                 if ds in mapping and ds in all_sigs]
    print(f"  可用数据集: {available}")

    # 获取特征列（从第一个有效签名中取）
    feature_cols = None
    for ds in available:
        labels = mapping[ds]
        matched = {l: all_sigs[ds][l] for l in labels if l in all_sigs[ds]}
        if matched:
            feature_cols = list(matched.values())[0]["feature_cols"]
            break
    if feature_cols is None:
        print("  无法获取特征列，跳过")
        return []

    # 加载所有需要的数据集
    print("  加载数据集...")
    datasets = {}
    for ds in available:
        df = load_dataset(data_dir, ds)
        if df is not None:
            datasets[ds] = df
            print(f"    {ds}: {len(df):,d} 行")

    rows = []

    # 对每个"目标数据集"进行实验
    for tgt_ds in available:
        if tgt_ds not in datasets:
            continue
        tgt_df     = datasets[tgt_ds]
        tgt_labels = mapping[tgt_ds]

        print(f"\n  目标数据集: {tgt_ds}  标签={tgt_labels}")

        # ── 条件1：Full-40（全特征，上界基线）──────────────────────────────
        X_tr, X_te, y_tr, y_te = build_binary_dataset(
            tgt_df, feature_cols, tgt_labels, RANDOM_SEED
        )
        if X_tr is None:
            print(f"    {tgt_ds} 无正例，跳过")
            continue

        metrics_full = run_xgb(X_tr, X_te, y_tr, y_te)
        print(f"    [Full-40]   F1={metrics_full['f1']:.4f}  "
              f"AUC={metrics_full['auc_roc']:.4f}")
        rows.append({
            "semantic_class": sem_cls,
            "source":         "—",
            "target":         tgt_ds,
            "condition":      "Full-40",
            "n_features":     40,
            "env_same":       True,
            **metrics_full,
        })

        # ── 条件4：Target-10（目标数据集自身 SHAP Top-10，理论上界）─────────
        tgt_topk = get_topk_features(all_sigs[tgt_ds], tgt_labels,
                                     feature_cols, TOP_K)
        if tgt_topk:
            tgt_idx  = [feature_cols.index(f) for f in tgt_topk]
            X_tr4, X_te4, y_tr4, y_te4 = build_binary_dataset(
                tgt_df, tgt_topk, tgt_labels, RANDOM_SEED
            )
            if X_tr4 is not None:
                m4 = run_xgb(X_tr4, X_te4, y_tr4, y_te4)
                print(f"    [Target-10] F1={m4['f1']:.4f}  "
                      f"AUC={m4['auc_roc']:.4f}  feats={tgt_topk[:3]}...")
                rows.append({
                    "semantic_class": sem_cls,
                    "source":         tgt_ds,
                    "target":         tgt_ds,
                    "condition":      "Target-10",
                    "n_features":     TOP_K,
                    "env_same":       True,
                    **m4,
                })

        # ── 条件3：Random-10（随机特征，下界基线）──────────────────────────
        random_f1s, random_aucs = [], []
        rng = np.random.default_rng(RANDOM_SEED)
        for rep in range(N_RANDOM_REPS):
            rand_feats = list(rng.choice(feature_cols, size=TOP_K,
                                         replace=False))
            X_tr3, X_te3, y_tr3, y_te3 = build_binary_dataset(
                tgt_df, rand_feats, tgt_labels, RANDOM_SEED + rep
            )
            if X_tr3 is not None:
                m3 = run_xgb(X_tr3, X_te3, y_tr3, y_te3)
                random_f1s.append(m3["f1"])
                random_aucs.append(m3["auc_roc"])

        rand_f1_mean  = float(np.mean(random_f1s))
        rand_auc_mean = float(np.mean(random_aucs))
        print(f"    [Random-10] F1={rand_f1_mean:.4f}±{np.std(random_f1s):.4f}  "
              f"AUC={rand_auc_mean:.4f}  (mean of {N_RANDOM_REPS} reps)")
        rows.append({
            "semantic_class": sem_cls,
            "source":         "Random",
            "target":         tgt_ds,
            "condition":      "Random-10",
            "n_features":     TOP_K,
            "env_same":       True,
            "f1":             round(rand_f1_mean, 4),
            "f1_std":         round(float(np.std(random_f1s)), 4),
            "auc_roc":        round(rand_auc_mean, 4),
            "n_pos_test":     metrics_full["n_pos_test"],
            "n_neg_test":     metrics_full["n_neg_test"],
        })

        # ── 条件2：SHAP-10（源数据集 SHAP Top-10 → 目标数据集）────────────
        for src_ds in available:
            if src_ds == tgt_ds or src_ds not in datasets:
                continue

            src_topk = get_topk_features(all_sigs[src_ds], mapping[src_ds],
                                         feature_cols, TOP_K)
            if not src_topk:
                continue

            # 检查特征列在目标数据集中是否存在
            valid = [f for f in src_topk if f in tgt_df.columns]
            if len(valid) < TOP_K:
                print(f"    {src_ds}→{tgt_ds}: 特征不完整 "
                      f"({len(valid)}/{TOP_K})，跳过")
                continue

            X_tr2, X_te2, y_tr2, y_te2 = build_binary_dataset(
                tgt_df, src_topk, tgt_labels, RANDOM_SEED
            )
            if X_tr2 is None:
                continue

            m2 = run_xgb(X_tr2, X_te2, y_tr2, y_te2)

            # Mann-Whitney U：SHAP-10 vs Random-10 的各 rep F1
            # 用单次 SHAP-10 值与 random_f1s 比较
            if random_f1s:
                _, pval_mw = stats.mannwhitneyu(
                    [m2["f1"]] * N_RANDOM_REPS,
                    random_f1s,
                    alternative="greater"
                )
            else:
                pval_mw = np.nan

            env_same = DS_ENV.get(src_ds) == DS_ENV.get(tgt_ds)
            pair_label = f"{src_ds}→{tgt_ds}"

            print(f"    [SHAP-10 {src_ds}→{tgt_ds}] "
                  f"F1={m2['f1']:.4f}  AUC={m2['auc_roc']:.4f}  "
                  f"env_same={'yes' if env_same else 'no'}  "
                  f"p(>random)={pval_mw:.4f}  "
                  f"feats={src_topk[:3]}...")

            rows.append({
                "semantic_class": sem_cls,
                "source":         src_ds,
                "target":         tgt_ds,
                "pair":           pair_label,
                "condition":      "SHAP-10",
                "n_features":     TOP_K,
                "env_same":       env_same,
                "env_src":        DS_ENV.get(src_ds, "?"),
                "env_tgt":        DS_ENV.get(tgt_ds, "?"),
                "src_topk_feats": str(src_topk),
                "mw_pval_vs_random": round(float(pval_mw), 6),
                **m2,
            })

    return rows


# ══════════════════════════════════════════════════════════════════════════════
# 结果可视化
# ══════════════════════════════════════════════════════════════════════════════
def plot_transfer_results(df: pd.DataFrame, out_dir: Path):
    """
    主结果图：对每个语义类别，展示四个条件下的 F1 比较。
    按"同环境"vs"跨环境"分组，突出 SHAP-10 的相对表现。
    """
    sem_classes = df["semantic_class"].unique()
    n_cls = len(sem_classes)

    fig, axes = plt.subplots(1, n_cls,
                             figsize=(STYLE["full_width"], 3.0),
                             sharey=False)
    if n_cls == 1:
        axes = [axes]

    cond_colors = {
        "Full-40":   "#2166ac",
        "Target-10": "#4dac26",
        "SHAP-10":   "#d6604d",
        "Random-10": "#aaaaaa",
    }
    cond_order = ["Full-40", "Target-10", "SHAP-10", "Random-10"]

    for ax, sem_cls in zip(axes, sem_classes):
        sub = df[df["semantic_class"] == sem_cls].copy()

        # 目标数据集列表
        tgt_datasets = [ds for ds in DS_ORDER
                        if ds in sub["target"].values]
        n_tgt = len(tgt_datasets)
        if n_tgt == 0:
            ax.set_visible(False)
            continue

        x_base = np.arange(n_tgt)
        width  = 0.18

        for ci, cond in enumerate(cond_order):
            cond_sub = sub[sub["condition"] == cond]
            if cond_sub.empty:
                continue

            # SHAP-10：按目标数据集取均值（可能有多个源）
            if cond == "SHAP-10":
                vals = []
                for tgt in tgt_datasets:
                    tgt_rows = cond_sub[cond_sub["target"] == tgt]
                    if not tgt_rows.empty:
                        # 同环境用实心，跨环境用斜线
                        same_env = tgt_rows[tgt_rows["env_same"]]
                        diff_env = tgt_rows[~tgt_rows["env_same"]]

                        if not same_env.empty:
                            ax.bar(x_base[tgt_datasets.index(tgt)] + ci * width,
                                   same_env["f1"].mean(), width=width,
                                   color=cond_colors[cond], alpha=0.85,
                                   zorder=2)
                        if not diff_env.empty:
                            ax.bar(x_base[tgt_datasets.index(tgt)] + ci * width,
                                   diff_env["f1"].mean(), width=width,
                                   color=cond_colors[cond], alpha=0.40,
                                   hatch="///", zorder=2)
                continue

            # 其他条件：逐目标数据集画条
            for ti, tgt in enumerate(tgt_datasets):
                tgt_rows = cond_sub[cond_sub["target"] == tgt]
                if tgt_rows.empty:
                    continue
                val = tgt_rows["f1"].iloc[0]
                ax.bar(ti + ci * width, val, width=width,
                       color=cond_colors[cond], alpha=0.85, zorder=2)

        ax.set_xticks(x_base + 1.5 * width)
        ax.set_xticklabels(tgt_datasets, fontsize=7)
        ax.set_ylabel("F1 score")
        ax.set_ylim(0, 1.08)
        ax.set_title(f"({chr(ord('a') + list(sem_classes).index(sem_cls))}) {sem_cls}",
                     loc="left", fontsize=8, fontweight="bold")
        ax.grid(axis="y", linewidth=0.3, alpha=0.4, zorder=0)
        ax.axhline(0.9, color="#888888", linestyle="--",
                   linewidth=0.6, alpha=0.6)

    # 图例
    handles = [
        mpatches.Patch(color=cond_colors[c], alpha=0.85, label=c)
        for c in cond_order
    ]
    handles.append(mpatches.Patch(color=cond_colors["SHAP-10"],
                                  alpha=0.85, hatch="///",
                                  label="SHAP-10 (cross-env)"))
    fig.legend(handles=handles, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.08),
               fontsize=7, frameon=False, columnspacing=1.0)

    fig.suptitle(
        "Figure 7. SHAP-guided feature transfer experiment\n"
        "(solid = same network environment, hatched = cross-environment)",
        fontsize=9, y=1.01
    )
    fig.tight_layout()
    stem = out_dir / "analysis2_transfer_figure"
    save_fig(fig, stem)
    print(f"  主结果图: {stem.name}.pdf/.png")


def plot_env_correlation(df: pd.DataFrame, out_dir: Path):
    """
    补充图：SHAP-10 F1 vs 环境相似度（同环境/跨环境分组散点）
    """
    shap_rows = df[df["condition"] == "SHAP-10"].copy()
    if shap_rows.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(STYLE["full_width"], 2.6))

    for ax_idx, metric in enumerate(["f1", "auc_roc"]):
        ax = axes[ax_idx]
        same = shap_rows[shap_rows["env_same"] == True][metric]
        diff = shap_rows[shap_rows["env_same"] == False][metric]

        # 箱线图 + 散点
        bp = ax.boxplot([same.values, diff.values],
                        positions=[0, 1], vert=True,
                        patch_artist=True, widths=0.35,
                        boxprops=dict(facecolor="#d1e5f0", linewidth=0.7),
                        medianprops=dict(color="#2166ac", linewidth=1.5),
                        whiskerprops=dict(linewidth=0.7),
                        capprops=dict(linewidth=0.7),
                        flierprops=dict(marker="x", markersize=3))

        jitter_s = np.random.default_rng(42).uniform(-0.08, 0.08, len(same))
        jitter_d = np.random.default_rng(43).uniform(-0.08, 0.08, len(diff))
        ax.scatter(np.zeros(len(same)) + jitter_s, same,
                   s=20, color="#2166ac", alpha=0.7, zorder=3)
        ax.scatter(np.ones(len(diff)) + jitter_d, diff,
                   s=20, color="#d6604d", alpha=0.7, zorder=3)

        # Mann-Whitney U 检验
        if len(same) > 0 and len(diff) > 0:
            _, pval = stats.mannwhitneyu(same, diff, alternative="greater")
            sig_str = f"p={pval:.3f}"
            if pval < 0.05:
                sig_str += " *"
            ax.text(0.5, 0.96, sig_str, ha="center", va="top",
                    transform=ax.transAxes, fontsize=7,
                    color="#333333")

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Same env.\n(IT↔IT or IoT↔IoT)",
                             "Cross env.\n(IT↔IoT)"], fontsize=7)
        ax.set_ylabel(metric.replace("_", "-").upper())
        ax.set_ylim(0, 1.08)
        ax.set_title(f"({'ab'[ax_idx]}) SHAP-10 {metric.upper()} "
                     f"by environment similarity",
                     loc="left", fontsize=8, fontweight="bold")
        ax.grid(axis="y", linewidth=0.3, alpha=0.4)

    fig.suptitle(
        "Figure 8. SHAP-guided transfer performance vs. network environment similarity",
        fontsize=9, y=1.01
    )
    fig.tight_layout()
    stem = out_dir / "analysis2_env_correlation"
    save_fig(fig, stem)
    print(f"  环境相关图: {stem.name}.pdf/.png")


# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════
def main():
    args = parse_args()
    rq1_dir  = Path(args.rq1_dir)
    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_style()
    np.random.seed(RANDOM_SEED)

    print(f"\n{'#'*60}")
    print(f"#  分析2：SHAP-guided 特征迁移实验（RQ3）")
    print(f"#  分析类别: {args.classes}")
    print(f"{'#'*60}\n")

    print("加载 RQ1 SHAP 签名...")
    all_sigs = load_shap_sigs(rq1_dir)
    print(f"已加载 {len(all_sigs)} 个数据集签名\n")

    all_rows = []

    for sem_cls in args.classes:
        if sem_cls not in SEMANTIC_MAPPING:
            print(f"'{sem_cls}' 不在映射表，跳过")
            continue
        mapping = SEMANTIC_MAPPING[sem_cls]
        rows = run_transfer_experiment(
            sem_cls, mapping, all_sigs, data_dir, out_dir
        )
        all_rows.extend(rows)

    if not all_rows:
        print("无结果，退出")
        return

    # ── 保存全量结果 ─────────────────────────────────────────────────────────
    result_df = pd.DataFrame(all_rows)
    result_csv = out_dir / "analysis2_transfer_results.csv"
    result_df.to_csv(result_csv, index=False)
    print(f"\n全量结果: {result_csv}")

    # ── 汇总统计 ─────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  实验结果汇总")
    print("="*60)

    shap_rows = result_df[result_df["condition"] == "SHAP-10"]
    full_rows = result_df[result_df["condition"] == "Full-40"]
    rand_rows = result_df[result_df["condition"] == "Random-10"]

    for sem_cls in args.classes:
        s = shap_rows[shap_rows["semantic_class"] == sem_cls]
        f = full_rows[full_rows["semantic_class"] == sem_cls]
        r = rand_rows[rand_rows["semantic_class"] == sem_cls]
        if s.empty:
            continue
        print(f"\n  {sem_cls}:")
        print(f"    Full-40   F1: {f['f1'].mean():.4f}")
        print(f"    SHAP-10 (同环境) F1: "
              f"{s[s['env_same']]['f1'].mean():.4f}"
              if not s[s["env_same"]].empty else "    SHAP-10 (同环境): 无")
        print(f"    SHAP-10 (跨环境) F1: "
              f"{s[~s['env_same']]['f1'].mean():.4f}"
              if not s[~s["env_same"]].empty else "    SHAP-10 (跨环境): 无")
        print(f"    Random-10 F1: {r['f1'].mean():.4f}")

    summary_csv = out_dir / "analysis2_transfer_summary.csv"
    result_df.groupby(["semantic_class", "condition", "env_same"])[
        ["f1", "auc_roc"]
    ].agg(["mean", "std"]).round(4).to_csv(summary_csv)
    print(f"\n汇总 CSV: {summary_csv}")

    # ── 绘图 ─────────────────────────────────────────────────────────────────
    print("\n绘制结果图...")
    plot_transfer_results(result_df, out_dir)
    plot_env_correlation(result_df, out_dir)

    print(f"\n分析2全部完成，输出目录: {out_dir.resolve()}\n")


if __name__ == "__main__":
    main()