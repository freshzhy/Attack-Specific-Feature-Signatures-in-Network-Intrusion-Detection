#!/usr/bin/env python3
"""
论文可视化图表生成脚本
第二篇NIDS论文 — 图表生成

生成以下五张论文图：
  Figure 1：RQ1 模型性能汇总（四数据集 F1 / AUC-ROC 分组柱状图）
  Figure 2：RQ1 SHAP 特征签名热力图（各数据集攻击类别 × 特征）
  Figure 3：RQ1 Top-10 特征签名气泡图（UNSW-NB15，展示方向与量级）
  Figure 4：RQ2 迁移性矩阵热力图（Jaccard + Spearman 并排）
  Figure 5：RQ2 共同特征 Upset 风格集合图（DoS 四数据集）

输出格式：PDF（论文嵌入用）+ PNG（预览用），300 DPI
输出目录：./results/figures/

用法：
  python plot_paper_figures.py --rq1-dir ./results/rq1/ --rq2-dir ./results/rq2/
  python plot_paper_figures.py --rq1-dir ./results/rq1/ --rq2-dir ./results/rq2/ --fig 1 2
"""

import argparse
import json
import pickle
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# 全局样式配置（IEEE Access / Computers & Security 投稿规范）
# ══════════════════════════════════════════════════════════════════════════════
STYLE = {
    "font_family":   "DejaVu Sans",
    "font_size":     8,
    "title_size":    9,
    "label_size":    8,
    "tick_size":     7,
    "legend_size":   7,
    "fig_dpi":       300,
    "col_width":     3.5,   # 单栏宽度（英寸），IEEE 双栏 = 3.5"
    "full_width":    7.16,  # 双栏满宽（英寸）
    # 颜色方案（色盲友好）
    "colors": {
        "blue":   "#2166ac",
        "red":    "#d6604d",
        "green":  "#4dac26",
        "orange": "#f4a582",
        "purple": "#7b2d8b",
        "gray":   "#888888",
        "light_blue": "#d1e5f0",
        "light_red":  "#fddbc7",
    },
    "dataset_colors": {
        "UNSW": "#2166ac",
        "CIC":  "#d6604d",
        "ToN":  "#4dac26",
        "BoT":  "#7b2d8b",
    },
    "dataset_markers": {
        "UNSW": "o",
        "CIC":  "s",
        "ToN":  "^",
        "BoT":  "D",
    },
}

DATASET_SHORT = {
    "NF-UNSW-NB15-v2":       "UNSW",
    "NF-CSE-CIC-IDS2018-v2": "CIC",
    "NF-ToN-IoT-v2":         "ToN",
    "NF-BoT-IoT-v2":         "BoT",
}

# 数据集显示顺序
DS_ORDER = ["UNSW", "CIC", "ToN", "BoT"]

OUT_DIR = Path("./results/figures")


def setup_style():
    plt.rcParams.update({
        "font.family":      STYLE["font_family"],
        "font.size":        STYLE["font_size"],
        "axes.titlesize":   STYLE["title_size"],
        "axes.labelsize":   STYLE["label_size"],
        "xtick.labelsize":  STYLE["tick_size"],
        "ytick.labelsize":  STYLE["tick_size"],
        "legend.fontsize":  STYLE["legend_size"],
        "axes.linewidth":   0.5,
        "axes.spines.top":  False,
        "axes.spines.right":False,
        "grid.linewidth":   0.4,
        "grid.alpha":       0.4,
        "figure.dpi":       STYLE["fig_dpi"],
        "savefig.dpi":      STYLE["fig_dpi"],
        "savefig.bbox":     "tight",
        "savefig.pad_inches": 0.02,
    })


def save_fig(fig, name: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUT_DIR / f"{name}.pdf"
    png_path = OUT_DIR / f"{name}.png"
    fig.savefig(pdf_path, format="pdf")
    fig.savefig(png_path, format="png")
    print(f"  {name}.pdf / .png")
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description="论文图表生成")
    p.add_argument("--rq1-dir", default="./results/rq1/")
    p.add_argument("--rq2-dir", default="./results/rq2/")
    p.add_argument("--fig", nargs="+", type=int, default=[1, 2, 3, 4, 5],
                   help="只生成指定编号的图（默认全部）")
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# 数据加载工具
# ══════════════════════════════════════════════════════════════════════════════
def load_metrics(rq1_dir: Path) -> dict[str, pd.DataFrame]:
    """加载四个数据集的模型性能指标"""
    result = {}
    for ds_full, ds_short in DATASET_SHORT.items():
        path = rq1_dir / f"{ds_full}_rq1_model_metrics.csv"
        if path.exists():
            result[ds_short] = pd.read_csv(path)
        else:
            print(f"  未找到: {path.name}")
    return result


def load_shap_matrices(rq1_dir: Path) -> dict[str, pd.DataFrame]:
    """加载四个数据集的 SHAP 矩阵（类别 × 特征）"""
    result = {}
    for ds_full, ds_short in DATASET_SHORT.items():
        path = rq1_dir / f"{ds_full}_rq1_shap_matrix.csv"
        if path.exists():
            df = pd.read_csv(path, index_col=0)
            result[ds_short] = df
        else:
            print(f"  未找到: {path.name}")
    return result


def load_topk(rq1_dir: Path) -> dict[str, pd.DataFrame]:
    """加载 Top-K 签名表"""
    result = {}
    for ds_full, ds_short in DATASET_SHORT.items():
        # 尝试 top10 或其他 K
        for k in [10, 5, 15]:
            path = rq1_dir / f"{ds_full}_rq1_top{k}_signatures.csv"
            if path.exists():
                result[ds_short] = pd.read_csv(path)
                break
    return result


def load_rq2(rq2_dir: Path) -> dict:
    """加载 RQ2 分析结果"""
    summary_path = rq2_dir / "rq2_summary.json"
    report_path  = rq2_dir / "rq2_transferability_report.csv"
    result = {}
    if summary_path.exists():
        with open(summary_path, encoding="utf-8") as f:
            result["summary"] = json.load(f)
    if report_path.exists():
        result["report"] = pd.read_csv(report_path)
    for cls in ["DoS", "DDoS", "Reconnaissance"]:
        for metric in ["jaccard", "spearman"]:
            p = rq2_dir / f"rq2_{metric}_{cls}.csv"
            if p.exists():
                result[f"{metric}_{cls}"] = pd.read_csv(p, index_col=0)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1：RQ1 模型性能汇总
# ══════════════════════════════════════════════════════════════════════════════
def plot_fig1_performance(metrics_by_ds: dict[str, pd.DataFrame]):
    """
    双面板：左=F1，右=AUC-ROC
    每个数据集的攻击类别按 F1 降序排列，用不同标记区分数据集
    """
    print("\n[Figure 1] RQ1 模型性能汇总")

    fig, axes = plt.subplots(1, 2, figsize=(STYLE["full_width"], 3.2))

    for ax_idx, (metric, xlabel) in enumerate([("f1", "F1 score"),
                                                ("auc_roc", "AUC-ROC")]):
        ax = axes[ax_idx]

        y_pos = 0
        yticks, ylabels, ycolors = [], [], []

        for ds in DS_ORDER:
            if ds not in metrics_by_ds:
                continue
            df = metrics_by_ds[ds].sort_values(metric, ascending=True)
            color  = STYLE["dataset_colors"][ds]
            marker = STYLE["dataset_markers"][ds]

            for _, row in df.iterrows():
                val = row[metric]
                ax.barh(y_pos, val, height=0.7,
                        color=color, alpha=0.75, zorder=2)
                ax.plot(val, y_pos, marker=marker,
                        color=color, markersize=4, zorder=3)

                # 数值标注（F1 < 0.7 才显示，避免挤压）
                if val < 0.85:
                    ax.text(val + 0.005, y_pos, f"{val:.3f}",
                            va="center", fontsize=6,
                            color=STYLE["colors"]["gray"])

                yticks.append(y_pos)
                ylabels.append(row["class"])
                ycolors.append(color)
                y_pos += 1

            # 数据集分隔线
            ax.axhline(y_pos - 0.5, color="white", linewidth=2, zorder=4)
            y_pos += 0.3

        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels, fontsize=6.5)
        for tick, col in zip(ax.get_yticklabels(), ycolors):
            tick.set_color(col)

        ax.set_xlabel(xlabel)
        ax.set_xlim(0.3, 1.05)
        ax.axvline(0.9, color=STYLE["colors"]["gray"],
                   linestyle="--", linewidth=0.6, alpha=0.6, zorder=1)
        ax.grid(axis="x", zorder=0)
        ax.set_title(f"({'a' if ax_idx == 0 else 'b'}) {xlabel}",
                     loc="left", fontweight="bold")

    # 图例
    legend_handles = [
        mpatches.Patch(color=STYLE["dataset_colors"][ds], label=ds)
        for ds in DS_ORDER if ds in metrics_by_ds
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=4, bbox_to_anchor=(0.5, -0.04),
               frameon=False, fontsize=STYLE["legend_size"])

    fig.suptitle("Figure 1. One-vs-rest classification performance across four NFv2 datasets",
                 fontsize=STYLE["title_size"], y=1.01)
    fig.tight_layout()
    save_fig(fig, "fig1_rq1_performance")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2：RQ1 SHAP 热力图（四数据集 2×2 布局）
# ══════════════════════════════════════════════════════════════════════════════
def plot_fig2_shap_heatmap(shap_matrices: dict[str, pd.DataFrame]):
    """
    四个子图（2×2），每个显示该数据集的 SHAP |均值| 热力图。
    只显示在该数据集中至少有一类进入 Top-10 的特征。
    """
    print("\n[Figure 2] SHAP 特征签名热力图")

    available = [ds for ds in DS_ORDER if ds in shap_matrices]
    n = len(available)
    ncols = 2
    nrows = (n + 1) // 2

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(STYLE["full_width"], nrows * 2.8))
    axes = np.array(axes).flatten()

    for i, ds in enumerate(available):
        ax = axes[i]
        mat = shap_matrices[ds]

        # 只保留至少一类中绝对值 > 阈值的特征
        threshold = mat.values.max() * 0.02
        feat_mask = (mat > threshold).any(axis=0)
        mat_filtered = mat.loc[:, feat_mask]

        # 对特征按最大 SHAP 值降序排列
        feat_order = mat_filtered.max(axis=0).sort_values(ascending=False).index
        mat_plot = mat_filtered[feat_order]

        im = ax.imshow(mat_plot.values, aspect="auto", cmap="YlOrRd",
                       vmin=0, vmax=mat_plot.values.max())

        ax.set_xticks(range(len(feat_order)))
        ax.set_xticklabels(feat_order, rotation=45, ha="right",
                           fontsize=5.5)
        ax.set_yticks(range(len(mat_plot.index)))
        ax.set_yticklabels(mat_plot.index, fontsize=6.5)

        # 在每个格子里标最大值列（★）
        max_col = mat_plot.values.argmax(axis=1)
        for row_idx, col_idx in enumerate(max_col):
            ax.text(col_idx, row_idx, "★", ha="center", va="center",
                    fontsize=5, color="white", alpha=0.9)

        panel = chr(ord("a") + i)
        ax.set_title(f"({panel}) {ds}", loc="left",
                     fontweight="bold", fontsize=STYLE["title_size"])

        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02,
                     label="|SHAP| mean")

    # 隐藏多余子图
    for j in range(len(available), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Figure 2. Attack-specific SHAP feature signatures across four NFv2 datasets\n"
        "★ = dominant feature per attack class",
        fontsize=STYLE["title_size"], y=1.01
    )
    fig.tight_layout()
    save_fig(fig, "fig2_rq1_shap_heatmap")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3：RQ1 气泡图（UNSW-NB15 Top-10，方向 + 量级）
# ══════════════════════════════════════════════════════════════════════════════
def plot_fig3_bubble(topk_by_ds: dict[str, pd.DataFrame]):
    """
    UNSW-NB15 的气泡图：
    x = 攻击类别，y = 特征名，气泡大小 = |SHAP|，颜色 = 方向（攻击↑/正常↓）
    展示签名的特异性和方向
    """
    print("\n[Figure 3] SHAP 气泡图（UNSW-NB15）")

    ds = "UNSW"
    if ds not in topk_by_ds:
        print(f"  {ds} 数据未找到，跳过")
        return

    df = topk_by_ds[ds].copy()

    # 特征按出现频次 + 平均 |SHAP| 排序（高频高重要度在上）
    feat_importance = df.groupby("feature")["abs_shap"].mean().sort_values(ascending=True)
    feat_order = feat_importance.index.tolist()
    class_order = df.groupby("class")["abs_shap"].mean().sort_values(ascending=False).index.tolist()

    feat_idx   = {f: i for i, f in enumerate(feat_order)}
    class_idx  = {c: i for i, c in enumerate(class_order)}

    x = [class_idx[r["class"]] for _, r in df.iterrows()]
    y = [feat_idx[r["feature"]] for _, r in df.iterrows()]
    sizes  = [r["abs_shap"] * 800 for _, r in df.iterrows()]
    colors = [STYLE["colors"]["blue"] if r["direction"] == "attack"
              else STYLE["colors"]["red"]
              for _, r in df.iterrows()]

    fig, ax = plt.subplots(figsize=(STYLE["full_width"], 3.8))

    sc = ax.scatter(x, y, s=sizes, c=colors, alpha=0.75,
                    edgecolors="white", linewidths=0.5, zorder=3)

    ax.set_xticks(range(len(class_order)))
    ax.set_xticklabels(class_order, rotation=30, ha="right",
                       fontsize=STYLE["tick_size"])
    ax.set_yticks(range(len(feat_order)))
    ax.set_yticklabels(feat_order, fontsize=6.5)

    ax.grid(True, linewidth=0.3, alpha=0.4, zorder=0)
    ax.set_xlim(-0.6, len(class_order) - 0.4)
    ax.set_ylim(-0.6, len(feat_order) - 0.4)

    # 图例：方向
    legend_dir = [
        mpatches.Patch(color=STYLE["colors"]["blue"],
                       alpha=0.75, label="↑ Towards attack"),
        mpatches.Patch(color=STYLE["colors"]["red"],
                       alpha=0.75, label="↓ Towards benign"),
    ]
    # 图例：气泡大小
    for size_val, label in [(0.5, "0.5"), (2.0, "2.0"), (4.0, "4.0")]:
        legend_dir.append(
            plt.scatter([], [], s=size_val * 800, c="gray",
                        alpha=0.5, label=f"|SHAP|={label}")
        )

    ax.legend(handles=legend_dir, loc="lower right",
              fontsize=STYLE["legend_size"], framealpha=0.9,
              ncol=2, columnspacing=0.8)

    ax.set_xlabel("Attack class")
    ax.set_ylabel("NetFlow feature")
    ax.set_title(
        "Figure 3. SHAP feature signatures for NF-UNSW-NB15-v2 "
        "(bubble size = |SHAP mean|, color = contribution direction)",
        fontsize=STYLE["title_size"], loc="left"
    )
    fig.tight_layout()
    save_fig(fig, "fig3_rq1_bubble_unsw")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4：RQ2 迁移性矩阵热力图
# ══════════════════════════════════════════════════════════════════════════════
def plot_fig4_transferability(rq2_data: dict):
    """
    三行（DoS / DDoS / Reconnaissance）× 两列（Jaccard / Spearman）
    共 6 个热力图子图
    """
    print("\n[Figure 4] RQ2 迁移性矩阵热力图")

    classes = ["DoS", "DDoS", "Reconnaissance"]
    metrics = [("jaccard", "Jaccard similarity", 0, 1, "Blues"),
               ("spearman", "Spearman ρ",        -1, 1, "RdBu")]

    n_cls = len([c for c in classes
                 if f"jaccard_{c}" in rq2_data])
    if n_cls == 0:
        print("  未找到 RQ2 矩阵数据，跳过")
        return

    fig, axes = plt.subplots(n_cls, 2,
                             figsize=(STYLE["col_width"] * 2 + 0.5,
                                      n_cls * 1.8 + 0.4))
    if n_cls == 1:
        axes = axes[np.newaxis, :]

    row = 0
    for cls in classes:
        key_j = f"jaccard_{cls}"
        key_s = f"spearman_{cls}"
        if key_j not in rq2_data:
            continue

        for col, (key, cbar_label, vmin, vmax, cmap) in enumerate(
            [(key_j, "Jaccard", 0, 1, "Blues"),
             (key_s, "Spearman ρ", -1, 1, "RdBu")]
        ):
            ax  = axes[row, col]
            mat = rq2_data[key]
            ds_labels = list(mat.columns)
            vals = mat.values.astype(float)

            im = ax.imshow(vals, cmap=cmap, vmin=vmin, vmax=vmax,
                           aspect="equal")

            ax.set_xticks(range(len(ds_labels)))
            ax.set_yticks(range(len(ds_labels)))
            ax.set_xticklabels(ds_labels, fontsize=7)
            ax.set_yticklabels(ds_labels, fontsize=7)

            # 数值标注
            for i in range(len(ds_labels)):
                for j in range(len(ds_labels)):
                    v = vals[i, j]
                    txt_color = "white" if abs(v) > 0.6 else "black"
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=6.5, color=txt_color,
                            fontweight="bold" if i != j else "normal")

            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                         label=cbar_label)

            panel = chr(ord("a") + row * 2 + col)
            suffix = " (IT vs IoT)" if cls == "DoS" else ""
            ax.set_title(f"({panel}) {cls} — {cbar_label}{suffix}",
                         fontsize=STYLE["label_size"], loc="left",
                         fontweight="bold")

        row += 1

    fig.suptitle(
        "Figure 4. Cross-dataset transferability of SHAP attack signatures",
        fontsize=STYLE["title_size"], y=1.01
    )
    fig.tight_layout()
    save_fig(fig, "fig4_rq2_transferability_matrix")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5：RQ2 共同特征集合图（DoS 四数据集）
# ══════════════════════════════════════════════════════════════════════════════
def plot_fig5_shared_features(rq2_data: dict, topk_by_ds: dict):
    """
    DoS 四数据集 Top-10 特征的集合对比图：
    左：每个数据集的 Top-10 特征水平条形图（按 Jaccard 交集标色）
    右：数据集两两交集数量的矩阵（类 Upset 图简化版）
    """
    print("\n[Figure 5] DoS 共同特征集合图")

    # 从 rq2_topk_DoS.csv 加载
    topk_path_candidates = [
        Path("./results/rq2/rq2_topk_DoS.csv"),
    ]
    topk_df = None
    for p in topk_path_candidates:
        if p.exists():
            topk_df = pd.read_csv(p)
            break

    if topk_df is None:
        print("  未找到 rq2_topk_DoS.csv，尝试从 rq1 重建")
        # 从 rq1 结果重建
        rows = []
        dos_labels = {
            "UNSW": "DoS",
            "CIC":  None,   # CIC 是合并的，从 pkl 需要特殊处理
            "ToN":  "dos",
            "BoT":  "DoS",
        }
        for ds, label in dos_labels.items():
            if ds in topk_by_ds and label:
                sub = topk_by_ds[ds][topk_by_ds[ds]["class"] == label]
                for _, r in sub.iterrows():
                    rows.append({"dataset": ds, "rank": r["rank"],
                                 "feature": r["feature"]})
        if rows:
            topk_df = pd.DataFrame(rows)
        else:
            print("  无法重建数据，跳过 Figure 5")
            return

    datasets = topk_df["dataset"].unique()
    feat_sets = {ds: set(topk_df[topk_df["dataset"] == ds]["feature"])
                 for ds in datasets}

    # 找所有特征的并集，按出现次数排序
    all_feats  = set().union(*feat_sets.values())
    feat_count = {f: sum(1 for s in feat_sets.values() if f in s)
                  for f in all_feats}
    feat_sorted = sorted(all_feats,
                         key=lambda f: (-feat_count[f], f))

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(STYLE["full_width"], max(3.5, len(feat_sorted) * 0.28)),
        gridspec_kw={"width_ratios": [3, 1.5]}
    )

    # ── 左图：特征出现矩阵 ──────────────────────────────────────────────────
    ds_list = sorted(datasets)
    colors_ds = [STYLE["dataset_colors"].get(d, "#888888") for d in ds_list]

    for y_idx, feat in enumerate(feat_sorted):
        cnt = feat_count[feat]
        # 背景条（灰色，全宽）
        ax_left.barh(y_idx, len(ds_list), height=0.7,
                     color="#f0f0f0", zorder=1)
        # 实际出现的数据集（彩色圆点）
        for x_idx, ds in enumerate(ds_list):
            if feat in feat_sets.get(ds, set()):
                ax_left.scatter(x_idx, y_idx,
                                s=80, color=colors_ds[x_idx],
                                marker=STYLE["dataset_markers"].get(ds, "o"),
                                zorder=3)
            else:
                ax_left.scatter(x_idx, y_idx,
                                s=30, color="#dddddd",
                                marker="o", zorder=2)

        # 出现次数标注
        ax_left.text(len(ds_list) + 0.15, y_idx,
                     f"n={cnt}", va="center", fontsize=6,
                     color=STYLE["colors"]["gray"])

    ax_left.set_yticks(range(len(feat_sorted)))
    ax_left.set_yticklabels(feat_sorted, fontsize=6.5)
    ax_left.set_xticks(range(len(ds_list)))
    ax_left.set_xticklabels(ds_list, fontsize=7)
    ax_left.set_xlim(-0.5, len(ds_list) + 0.8)
    ax_left.set_ylim(-0.6, len(feat_sorted) - 0.4)
    ax_left.set_xlabel("Dataset")
    ax_left.set_title("(a) DoS Top-10 feature membership across datasets",
                       loc="left", fontsize=STYLE["label_size"],
                       fontweight="bold")
    ax_left.invert_yaxis()

    # 图例
    legend_h = [
        plt.scatter([], [], s=80,
                    color=STYLE["dataset_colors"].get(d, "#888"),
                    marker=STYLE["dataset_markers"].get(d, "o"),
                    label=d)
        for d in ds_list
    ]
    ax_left.legend(handles=legend_h, loc="lower right",
                   fontsize=STYLE["legend_size"], framealpha=0.9)

    # ── 右图：两两交集大小矩阵 ──────────────────────────────────────────────
    n = len(ds_list)
    inter_mat = np.zeros((n, n))
    for i, ds_a in enumerate(ds_list):
        for j, ds_b in enumerate(ds_list):
            if i == j:
                inter_mat[i, j] = len(feat_sets.get(ds_a, set()))
            else:
                inter_mat[i, j] = len(
                    feat_sets.get(ds_a, set()) & feat_sets.get(ds_b, set())
                )

    im = ax_right.imshow(inter_mat, cmap="Blues",
                         vmin=0, vmax=10, aspect="equal")
    ax_right.set_xticks(range(n))
    ax_right.set_yticks(range(n))
    ax_right.set_xticklabels(ds_list, fontsize=7)
    ax_right.set_yticklabels(ds_list, fontsize=7)

    for i in range(n):
        for j in range(n):
            v = int(inter_mat[i, j])
            txt_color = "white" if v > 6 else "black"
            ax_right.text(j, i, str(v), ha="center", va="center",
                          fontsize=8, fontweight="bold", color=txt_color)

    plt.colorbar(im, ax=ax_right, fraction=0.046, pad=0.04,
                 label="Shared features")
    ax_right.set_title("(b) Pairwise intersection\nsize (out of 10)",
                        loc="left", fontsize=STYLE["label_size"],
                        fontweight="bold")

    fig.suptitle(
        "Figure 5. DoS Top-10 feature overlap across four NFv2 datasets",
        fontsize=STYLE["title_size"], y=1.01
    )
    fig.tight_layout()
    save_fig(fig, "fig5_rq2_dos_shared_features")


# ══════════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════════
def main():
    args = parse_args()
    rq1_dir = Path(args.rq1_dir)
    rq2_dir = Path(args.rq2_dir)

    setup_style()

    print(f"\n{'#'*60}")
    print(f"#  论文图表生成")
    print(f"#  RQ1: {rq1_dir}")
    print(f"#  RQ2: {rq2_dir}")
    print(f"#  生成图: {args.fig}")
    print(f"{'#'*60}")

    # 按需加载数据
    need_rq1 = any(f in args.fig for f in [1, 2, 3])
    need_rq2 = any(f in args.fig for f in [4, 5])

    metrics_by_ds  = load_metrics(rq1_dir)  if need_rq1 else {}
    shap_matrices  = load_shap_matrices(rq1_dir) if need_rq1 else {}
    topk_by_ds     = load_topk(rq1_dir) if need_rq1 or 5 in args.fig else {}
    rq2_data       = load_rq2(rq2_dir)  if need_rq2 else {}

    # 生成各图
    if 1 in args.fig:
        plot_fig1_performance(metrics_by_ds)
    if 2 in args.fig:
        plot_fig2_shap_heatmap(shap_matrices)
    if 3 in args.fig:
        plot_fig3_bubble(topk_by_ds)
    if 4 in args.fig:
        plot_fig4_transferability(rq2_data)
    if 5 in args.fig:
        plot_fig5_shared_features(rq2_data, topk_by_ds)

    print(f"\n全部完成，输出目录: {OUT_DIR.resolve()}\n")


if __name__ == "__main__":
    main()
