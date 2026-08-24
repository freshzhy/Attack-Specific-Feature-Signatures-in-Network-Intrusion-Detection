#!/usr/bin/env python3
"""
NFv2 数据集批量探查脚本（轻量版）
第二篇NIDS论文 — 数据预处理阶段 Step 1b

与原版的区别：
  - 去掉耗时的统计摘要（describe + skew），大数据集跑太慢
  - 重复记录改为 10万行采样估算，不全量扫描
  - 支持 --out 将结果同时写入文件（终端 + 文件双输出）
  - 数据集名称自动从文件名提取

用法（单个）：
  python explore_nfv2_batch.py --data ./NF-CSE-CIC-IDS2018-v2/data/NF-CSE-CIC-IDS2018-v2.csv

用法（批量，用 batch_explore.sh）：
  bash batch_explore.sh
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd


# NFv2 标准特征集（Sarhan et al. 2022, Table 3）
# 43 个特征 = 41个流特征 + IPV4_SRC_ADDR + IPV4_DST_ADDR
NFV2_EXPECTED_FEATURES = {
    "L4_SRC_PORT", "L4_DST_PORT", "PROTOCOL", "L7_PROTO",
    "IN_BYTES", "IN_PKTS", "OUT_BYTES", "OUT_PKTS",
    "TCP_FLAGS", "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS",
    "FLOW_DURATION_MILLISECONDS", "DURATION_IN", "DURATION_OUT",
    "MIN_TTL", "MAX_TTL", "LONGEST_FLOW_PKT", "SHORTEST_FLOW_PKT",
    "MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN", "SRC_TO_DST_SECOND_BYTES",
    "DST_TO_SRC_SECOND_BYTES", "RETRANSMITTED_IN_BYTES",
    "RETRANSMITTED_IN_PKTS", "RETRANSMITTED_OUT_BYTES",
    "RETRANSMITTED_OUT_PKTS", "SRC_TO_DST_AVG_THROUGHPUT",
    "DST_TO_SRC_AVG_THROUGHPUT", "NUM_PKTS_UP_TO_128_BYTES",
    "NUM_PKTS_128_TO_256_BYTES", "NUM_PKTS_256_TO_512_BYTES",
    "NUM_PKTS_512_TO_1024_BYTES", "NUM_PKTS_1024_TO_1514_BYTES",
    "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT", "ICMP_TYPE", "ICMP_IPV4_TYPE",
    "DNS_QUERY_ID", "DNS_QUERY_TYPE", "DNS_TTL_ANSWER",
    "FTP_COMMAND_RET_CODE", "IPV4_SRC_ADDR", "IPV4_DST_ADDR",
}

SMALL_CLASS_THRESHOLD = 5000   # 小样本警告阈值
DUP_SAMPLE_SIZE       = 100_000  # 重复记录采样行数


def parse_args():
    parser = argparse.ArgumentParser(description="NFv2 批量探查（轻量版）")
    parser.add_argument("--data", required=True, help="CSV 文件路径")
    parser.add_argument("--out",  default=None,
                        help="结果输出文件路径（不指定则只打印到终端）")
    return parser.parse_args()


class Tee:
    """同时写入终端和文件的输出流"""
    def __init__(self, filepath: Path):
        self.terminal = sys.stdout
        self.file = open(filepath, "w", encoding="utf-8")

    def write(self, msg):
        self.terminal.write(msg)
        self.file.write(msg)

    def flush(self):
        self.terminal.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def section(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# ── 1. 字段清单 ───────────────────────────────────────────────────────────────
def check_schema(df: pd.DataFrame, dataset_name: str):
    section(f"[{dataset_name}]  1. 字段清单（共 {df.shape[1]} 列）")

    label_cols   = [c for c in df.columns if c.lower() in ("label", "attack")]
    feature_cols = [c for c in df.columns if c not in label_cols]

    print(f"标签列:   {label_cols}")
    print(f"特征列数: {len(feature_cols)}  （预期 43）")

    if len(feature_cols) != 43:
        print("特征列数与 NFv2 规范不符！")
    else:
        print("特征列数与 Sarhan NFv2 规范一致")

    # 列名对照
    actual_features = set(feature_cols)
    missing = NFV2_EXPECTED_FEATURES - actual_features
    extra   = actual_features - NFV2_EXPECTED_FEATURES

    if missing:
        print(f"预期但缺失: {sorted(missing)}")
    if extra:
        print(f"额外列（不在规范中）: {sorted(extra)}")
    if not missing and not extra:
        print("列名与 NFv2 规范完全匹配")

    # 数据类型异常（非数值的非标签列）
    non_num_feat = df[feature_cols].select_dtypes(exclude=[np.number]).columns.tolist()
    # IP 地址列是 str，属于预期，不算异常
    unexpected_str = [c for c in non_num_feat
                      if c not in ("IPV4_SRC_ADDR", "IPV4_DST_ADDR")]
    if unexpected_str:
        print(f"意外的非数值特征列: {unexpected_str}")
    else:
        print("特征列数据类型正常")

    return feature_cols, label_cols


# ── 2. 攻击类别分布 ───────────────────────────────────────────────────────────
def check_labels(df: pd.DataFrame, dataset_name: str):
    section(f"[{dataset_name}]  2. 攻击类别分布")

    # 自动识别列名
    attack_col = next((c for c in df.columns
                       if c.lower() in ("attack", "attack_type", "attack_cat")), None)
    label_col  = next((c for c in df.columns
                       if c.lower() == "label"), None)
    total = len(df)

    if label_col:
        print(f"\n二分类 '{label_col}':")
        for val, cnt in df[label_col].value_counts().sort_index().items():
            print(f"  {str(val):>6s}: {cnt:>12,d}  ({cnt/total*100:6.2f}%)")

    if attack_col:
        print(f"\n多分类 '{attack_col}':")
        vc = df[attack_col].value_counts().sort_values(ascending=False)
        for val, cnt in vc.items():
            flag = "  ← < 5000" if cnt < SMALL_CLASS_THRESHOLD else ""
            print(f"  {str(val):>25s}: {cnt:>12,d}  ({cnt/total*100:6.2f}%){flag}")
        print(f"  {'─'*25}  {'─'*12}")
        print(f"  {'Total':>25s}: {total:>12,d}")

        small = vc[vc < SMALL_CLASS_THRESHOLD]
        if len(small):
            print(f"\n小样本类别（< {SMALL_CLASS_THRESHOLD:,}）: "
                  f"{list(small.index)}  → 考虑排除")
    else:
        print("未找到多分类标签列，请检查列名")

    return attack_col


# ── 3. 缺失值 / Inf / 重复（采样）────────────────────────────────────────────
def check_quality(df: pd.DataFrame, dataset_name: str):
    section(f"[{dataset_name}]  3. 数据质量检查")

    # 缺失值
    total_null = df.isnull().sum().sum()
    print(f"缺失值: {total_null:,d}", end="")
    if total_null == 0:
        print("  done")
    else:
        print()
        bad = df.isnull().sum()
        bad = bad[bad > 0].sort_values(ascending=False)
        for col, n in bad.items():
            print(f"  {col}: {n:,d}  ({n/len(df)*100:.3f}%)")

    # Inf（只检查数值列）
    num_cols   = df.select_dtypes(include=[np.number]).columns
    total_inf  = np.isinf(df[num_cols].values).sum()
    print(f"Inf 值:  {total_inf:,d}", end="")
    if total_inf == 0:
        print("  done")
    else:
        print()
        inf_by_col = pd.Series(
            np.isinf(df[num_cols]).sum().values, index=num_cols
        )
        for col, n in inf_by_col[inf_by_col > 0].items():
            print(f"  {col}: {n:,d}")

    # 重复行（采样估算，大数据集全量扫描太慢）
    n_total  = len(df)
    n_sample = min(DUP_SAMPLE_SIZE, n_total)
    sample   = df.sample(n=n_sample, random_state=42)
    n_dup_sample = sample.duplicated().sum()
    dup_rate = n_dup_sample / n_sample
    dup_est  = int(dup_rate * n_total)

    print(f"重复行:  采样 {n_sample:,d} 行，"
          f"发现 {n_dup_sample} 条重复  "
          f"→ 全量估算 ≈ {dup_est:,d}  ({dup_rate*100:.2f}%)")
    if dup_est == 0:
        print("         无重复（估算）")


# ── 4. 规模与兼容性摘要 ───────────────────────────────────────────────────────
def check_scale(df: pd.DataFrame, dataset_name: str,
                feature_cols: list, attack_col: str):
    section(f"[{dataset_name}]  4. 规模与兼容性摘要")

    mem_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
    print(f"行数:     {len(df):>12,d}")
    print(f"列数:     {df.shape[1]:>12d}  （特征 {len(feature_cols)} + 标签 2）")
    print(f"内存:     {mem_mb:>11.1f} MB")

    # 排除 IP 列后的实验特征数
    exp_feat = [c for c in feature_cols
                if c not in ("IPV4_SRC_ADDR", "IPV4_DST_ADDR")]
    print(f"实验特征: {len(exp_feat):>12d}  （排除 IP 标识符后）")

    # MAX_IP_PKT_LEN 与 LONGEST_FLOW_PKT 重复列快速验证（采样）
    if "MAX_IP_PKT_LEN" in df.columns and "LONGEST_FLOW_PKT" in df.columns:
        sample = df[["MAX_IP_PKT_LEN", "LONGEST_FLOW_PKT"]].sample(
            n=min(50_000, len(df)), random_state=42
        )
        n_diff = (sample["MAX_IP_PKT_LEN"] != sample["LONGEST_FLOW_PKT"]).sum()
        if n_diff == 0:
            print("重复列:   MAX_IP_PKT_LEN ≡ LONGEST_FLOW_PKT（采样验证）→ 可删除一列")
        else:
            print(f"重复列:   MAX_IP_PKT_LEN ≠ LONGEST_FLOW_PKT（{n_diff} 行不同）→ 保留两列")

    # DoS 类别名称（跨数据集对比锚点）
    if attack_col:
        all_classes = sorted(df[attack_col].unique())
        dos_related = [c for c in all_classes
                       if "dos" in c.lower() or "ddos" in c.lower()]
        print(f"\n所有攻击类别: {all_classes}")
        print(f"DoS/DDoS 相关类别（跨数据集锚点）: "
              f"{dos_related if dos_related else '未检测到，请人工核对'}")


# ── 主流程 ────────────────────────────────────────────────────────────────────
def main():
    args     = parse_args()
    data_path = Path(args.data)

    if not data_path.exists():
        print(f"错误: 文件不存在 → {data_path}")
        sys.exit(1)

    dataset_name = data_path.stem   # e.g. "NF-CSE-CIC-IDS2018-v2"

    # 设置双输出
    tee = None
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tee = Tee(out_path)
        sys.stdout = tee

    try:
        print(f"\n{'#'*70}")
        print(f"#  数据集: {dataset_name}")
        print(f"#  文件:   {data_path}")
        print(f"#  大小:   {data_path.stat().st_size / 1024 / 1024:.1f} MB")
        print(f"{'#'*70}")

        print("正在读取（大文件请耐心等待）...")
        df = pd.read_csv(data_path, low_memory=False)
        print(f"读取完成: {df.shape[0]:,d} 行 × {df.shape[1]} 列")

        feature_cols, label_cols = check_schema(df, dataset_name)
        attack_col = check_labels(df, dataset_name)
        check_quality(df, dataset_name)
        check_scale(df, dataset_name, feature_cols, attack_col)

        section(f"[{dataset_name}]  探查完成")

    finally:
        if tee:
            sys.stdout = tee.terminal
            tee.close()
            print(f"结果已写入: {args.out}")


if __name__ == "__main__":
    main()
