#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# batch_preprocess.sh  —  对四个 NFv2 数据集依次运行预处理管道
#
# 用法（从仓库根目录 upload/ 运行）：
#   bash code/batch_preprocess.sh
#
# 注意：
#   - UNSW-NB15 已在上一步处理完毕，此处仍列入以保证日志完整
#   - 其余三个大数据集使用 --skip-dup-check（已通过采样验证）
#   - 日志实时打印到终端，同时保存到 ./preprocess_logs/
#   - [upload 版本改动] SCRIPT 路径改为 ./code/preprocess_nfv2.py，
#     以配合本仓库把脚本放在 code/ 子目录、数据/输出目录留在仓库根目录的结构
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

DATA_ROOT="."
OUT_DIR="./processed"
LOG_DIR="./preprocess_logs"
SCRIPT="./code/preprocess_nfv2.py"

mkdir -p "$LOG_DIR"

# UNSW-NB15：数据量小，做完整重复列验证
# 其余三个：数据量大，跳过逐行验证（已预验证），用 case 兼容 macOS bash 3.2
get_extra_args() {
    case "$1" in
        "NF-UNSW-NB15-v2")         echo "" ;;
        "NF-CSE-CIC-IDS2018-v2")   echo "--skip-dup-check" ;;
        "NF-ToN-IoT-v2")           echo "--skip-dup-check" ;;
        "NF-BoT-IoT-v2")           echo "--skip-dup-check" ;;
        *)                         echo "" ;;
    esac
}

DATASETS=(
    "NF-UNSW-NB15-v2"
    "NF-CSE-CIC-IDS2018-v2"
    "NF-ToN-IoT-v2"
    "NF-BoT-IoT-v2"
)

TOTAL=${#DATASETS[@]}

echo "======================================================================"
echo "  NFv2 批量预处理  —  共 ${TOTAL} 个数据集"
echo "  输出目录: $OUT_DIR"
echo "  日志目录: $LOG_DIR"
echo "======================================================================"

GRAND_START=$(date +%s)
IDX=0

for DS in "${DATASETS[@]}"; do
    IDX=$((IDX + 1))
    CSV_PATH="${DATA_ROOT}/${DS}/data/${DS}.csv"
    LOG_FILE="${LOG_DIR}/${DS}_preprocess.log"
    EXTRA=$(get_extra_args "$DS")

    echo ""
    echo "[${IDX}/${TOTAL}]  $DS"
    echo "       CSV : $CSV_PATH"
    echo "       日志: $LOG_FILE"
    echo "       参数: --out $OUT_DIR $EXTRA"

    if [ ! -f "$CSV_PATH" ]; then
        echo "       文件不存在，跳过！"
        continue
    fi

    START=$(date +%s)

    # tee：终端实时显示 + 写入日志
    python3 "$SCRIPT" \
        --data "$CSV_PATH" \
        --out  "$OUT_DIR"  \
        $EXTRA \
        2>&1 | tee "$LOG_FILE"

    END=$(date +%s)
    echo ""
    echo "       完成，耗时 $((END - START)) 秒"
    echo "----------------------------------------------------------------------"
done

GRAND_END=$(date +%s)

echo ""
echo "======================================================================"
echo "  全部完成。总耗时: $((GRAND_END - GRAND_START)) 秒"
echo ""
echo "  processed/ 目录："
ls -lh "$OUT_DIR"/ 2>/dev/null | grep -E "\.(csv|txt)$" || echo "  （无文件）"
echo "======================================================================"