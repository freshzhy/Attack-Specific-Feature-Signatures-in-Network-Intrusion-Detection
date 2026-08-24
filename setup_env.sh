#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# setup_env.sh  —  为第二篇NIDS论文创建 Python 虚拟环境并安装依赖
#
# 用法：
#   bash setup_env.sh
#
# 完成后激活环境：
#   source .venv/bin/activate
#
# 之后所有实验脚本在此环境下运行：
#   python shap_signatures_rq1.py --data ...
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

VENV_DIR=".venv"
PYTHON="python3"

echo "======================================================================"
echo "  NIDS 论文实验环境配置"
echo "======================================================================"

# ── 1. 检查 Python 版本 ───────────────────────────────────────────────────────
echo ""
echo "[1/4] 检查 Python 版本..."
PY_VERSION=$($PYTHON --version 2>&1)
echo "      $PY_VERSION"

# ── 2. 创建虚拟环境 ───────────────────────────────────────────────────────────
echo ""
echo "[2/4] 创建虚拟环境: $VENV_DIR ..."

if [ -d "$VENV_DIR" ]; then
    echo "      已存在，跳过创建（如需重建请先 rm -rf $VENV_DIR）"
else
    $PYTHON -m venv "$VENV_DIR"
    echo "      创建完成"
fi

# 激活
source "$VENV_DIR/bin/activate"
echo "      已激活: $(which python)"

# ── 3. 升级 pip ───────────────────────────────────────────────────────────────
echo ""
echo "[3/4] 升级 pip..."
pip install --upgrade pip --quiet
echo "      pip $(pip --version | awk '{print $2}')"

# ── 4. 安装依赖 ───────────────────────────────────────────────────────────────
echo ""
echo "[4/4] 安装实验依赖..."
echo "      （首次安装约需 3-5 分钟，视网络速度而定）"
echo ""

pip install \
    "numpy>=1.24" \
    "pandas>=2.0" \
    "scikit-learn>=1.3" \
    "xgboost>=2.0" \
    "shap>=0.44" \
    "matplotlib>=3.7" \
    "seaborn>=0.12"

echo ""
echo "======================================================================"
echo "  安装完成，版本确认："
echo "======================================================================"
python -c "
import numpy, pandas, sklearn, xgboost, shap, matplotlib, seaborn
print(f'  numpy       {numpy.__version__}')
print(f'  pandas      {pandas.__version__}')
print(f'  scikit-learn {sklearn.__version__}')
print(f'  xgboost     {xgboost.__version__}')
print(f'  shap        {shap.__version__}')
print(f'  matplotlib  {matplotlib.__version__}')
print(f'  seaborn     {seaborn.__version__}')
"

echo ""
echo "======================================================================"
echo "  使用方法："
echo ""
echo "  # 每次新开终端时激活环境："
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "  # 运行实验脚本："
echo "  python shap_signatures_rq1.py --data ./processed/NF-UNSW-NB15-v2_processed.csv --classes DoS Backdoor"
echo ""
echo "  # 退出虚拟环境："
echo "  deactivate"
echo "======================================================================"
