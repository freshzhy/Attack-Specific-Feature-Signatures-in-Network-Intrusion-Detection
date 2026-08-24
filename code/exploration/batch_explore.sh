#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# batch_explore.sh  —  对三个 NFv2 数据集依次运行探查脚本
#
# 用法（从仓库根目录 upload/ 运行）：
#   bash code/exploration/batch_explore.sh
#
# 前提：
#   1. 按下方 DATA_ROOT 设置数据集根目录
#   2. 三个数据集子目录名称与 DATASETS 数组一致
#   [upload 版本改动] SCRIPT 路径改为 ./code/exploration/explore_nfv2_batch.py，
#   以配合本仓库把脚本放在 code/exploration/ 子目录、数据目录留在仓库根目录的结构
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── 配置区（按实际路径修改） ──────────────────────────────────────────────────
DATA_ROOT="."          # 数据集根目录（相对或绝对路径均可）
OUT_DIR="./explore_results"   # 探查结果输出目录
SCRIPT="./code/exploration/explore_nfv2_batch.py"

# 三个待探查的数据集（目录名 = CSV 文件名，无后缀）
DATASETS=(
    "NF-CSE-CIC-IDS2018-v2"
    "NF-ToN-IoT-v2"
    "NF-BoT-IoT-v2"
)
# ─────────────────────────────────────────────────────────────────────────────

mkdir -p "$OUT_DIR"

echo "======================================================================"
echo "  NFv2 批量探查  —  共 ${#DATASETS[@]} 个数据集"
echo "  结果将保存到: $OUT_DIR"
echo "======================================================================"

TOTAL=${#DATASETS[@]}
IDX=0

for DS in "${DATASETS[@]}"; do
    IDX=$((IDX + 1))
    CSV_PATH="${DATA_ROOT}/${DS}/data/${DS}.csv"
    OUT_FILE="${OUT_DIR}/${DS}_explore.txt"

    echo ""
    echo "[${IDX}/${TOTAL}] 开始探查: $DS"
    echo "      CSV : $CSV_PATH"
    echo "      输出: $OUT_FILE"

    if [ ! -f "$CSV_PATH" ]; then
        echo "      文件不存在，跳过！请检查路径: $CSV_PATH"
        continue
    fi

    START=$(date +%s)

    python3 "$SCRIPT" \
        --data "$CSV_PATH" \
        --out  "$OUT_FILE"

    END=$(date +%s)
    echo "      完成，耗时 $((END - START)) 秒"
done

echo ""
echo "======================================================================"
echo "  全部完成。结果文件列表："
ls -lh "$OUT_DIR"/*.txt 2>/dev/null || echo "  （无输出文件）"
echo "======================================================================"
