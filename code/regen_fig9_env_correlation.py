#!/usr/bin/env python3
"""
重新生成 Fig. 9（环境相似度 vs 迁移性能）：
  - 旧图 (results/analysis2/analysis2_env_correlation.png) 用的是
    39 行的单次/小样本 analysis2_transfer_results.csv，且用单侧
    Mann-Whitney U（alternative="greater"），得到 p=0.859 (F1) / p=0.788 (AUC)，
    与正文 4.3.3 / Abstract / Conclusion 报告的 configuration-level
    two-sided Mann-Whitney U, p=0.335 (F1) 完全对不上；caption 还错误地
    写成 "Wilcoxon signed-rank"。
  - 新图改用与正文完全一致的口径：results/analysis3/analysis3_raw_results.csv
    （30-seed repeated evaluation, condition == "SHAP-10"），按
    (semantic_class, source, target, env_same) 分组取 30-seed 均值，
    得到 n=6 (same-env) / n=12 (cross-env) 的 configuration-level 均值，
    两侧 Mann-Whitney U 检验，与正文数字（F1: 0.908 vs 0.938, p=0.335；
    AUC-ROC 同法计算）完全一致。
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STYLE = {"full_width": 7.16, "fig_dpi": 300, "font_size": 8}

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

raw = pd.read_csv("results/analysis3/analysis3_raw_results.csv")
shap_rows = raw[raw["condition"] == "SHAP-10"].copy()
grp = shap_rows.groupby(["semantic_class", "source", "target", "env_same"], as_index=False).agg(
    f1_mean=("f1", "mean"), auc_mean=("auc_roc", "mean"), n_seeds=("seed", "count"))
assert len(grp) == 18, f"expected 18 configurations, got {len(grp)}"

out_dir = Path("results/analysis3")
out_dir.mkdir(parents=True, exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(STYLE["full_width"], 2.6))

for ax_idx, (metric, col) in enumerate([("F1", "f1_mean"), ("AUC-ROC", "auc_mean")]):
    ax = axes[ax_idx]
    same = grp[grp.env_same == True][col]
    diff = grp[grp.env_same == False][col]
    assert len(same) == 6 and len(diff) == 12

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
               s=24, color="#2166ac", alpha=0.75, zorder=3)
    ax.scatter(np.ones(len(diff)) + jitter_d, diff,
               s=24, color="#d6604d", alpha=0.75, zorder=3)

    _, pval = stats.mannwhitneyu(same, diff, alternative="two-sided")
    sig_str = f"Mann–Whitney U, p = {pval:.3f}"
    if pval < 0.05:
        sig_str += " *"
    ax.text(0.5, 0.96, sig_str, ha="center", va="top",
            transform=ax.transAxes, fontsize=6.5, color="#333333")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Same env.\n(IT↔IT or IoT↔IoT)",
                         "Cross env.\n(IT↔IoT)"], fontsize=7)
    ax.set_ylabel(metric)
    ax.set_ylim(0, 1.08)
    ax.set_title(f"({'ab'[ax_idx]}) SHAP-10 {metric} by environment similarity",
                 loc="left", fontsize=8, fontweight="bold")
    ax.grid(axis="y", linewidth=0.3, alpha=0.4)

    print(f"{metric}: same-env mean={same.mean():.4f} (n={len(same)}), "
          f"cross-env mean={diff.mean():.4f} (n={len(diff)}), "
          f"Mann-Whitney U two-sided p={pval:.4f}")

fig.suptitle(
    "Fig. 9. SHAP-guided transfer performance vs. network environment similarity\n"
    "(configuration-level mean over 30 seeds; n = 6 same-env., n = 12 cross-env. configurations)",
    fontsize=8.5, y=1.08
)
fig.tight_layout()
stem = out_dir / "analysis3_fig9_env_correlation"
fig.savefig(str(stem) + ".pdf", format="pdf")
fig.savefig(str(stem) + ".png", format="png")
plt.close(fig)
print(f"\nsaved {stem}.pdf / .png")
