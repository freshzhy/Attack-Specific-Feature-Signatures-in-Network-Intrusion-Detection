#!/usr/bin/env python3
"""
NF-UNSW-NB15-v2 数据探查脚本
第二篇NIDS论文 — 数据预处理阶段 Step 1

用途：验证数据集结构、攻击分布、质量问题，确认与实验管道的兼容性。
运行：python explore_nf_unsw_nb15_v2.py --data /path/to/NF-UNSW-NB15-v2.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="NF-UNSW-NB15-v2 数据探查"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="NF-UNSW-NB15-v2.csv 文件路径",
    )
    return parser.parse_args()


def section(title: str):
    """打印带分隔线的段落标题"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def explore_schema(df: pd.DataFrame):
    """1. 字段名、数据类型、非空计数"""
    section("1. 字段清单与数据类型（共 {} 列）".format(df.shape[1]))

    schema = pd.DataFrame({
        "Column": df.columns,
        "Dtype": df.dtypes.values,
        "Non-Null": df.notnull().sum().values,
        "Null": df.isnull().sum().values,
        "Unique": df.nunique().values,
    })
    schema.index = range(1, len(schema) + 1)
    schema.index.name = "#"
    print(schema.to_string())

    # 预期43特征 + Label + Attack 的检查
    expected_feature_count = 43
    label_cols = [c for c in df.columns if c.lower() in ("label", "attack")]
    feature_cols = [c for c in df.columns if c not in label_cols]
    print(f"\n标签列: {label_cols}")
    print(f"特征列数量: {len(feature_cols)}  (预期 {expected_feature_count})")

    if len(feature_cols) != expected_feature_count:
        print(f"特征列数与预期不符，请检查！")
    else:
        print("特征列数量与 Sarhan NFv2 规范一致")


def explore_labels(df: pd.DataFrame):
    """2. 攻击类别分布"""
    section("2. 攻击类别分布")

    # 尝试多种可能的列名
    attack_col = None
    for candidate in ["Attack", "attack", "ATTACK", "Attack_type",
                       "attack_cat", "Attack_cat"]:
        if candidate in df.columns:
            attack_col = candidate
            break

    label_col = None
    for candidate in ["Label", "label", "LABEL"]:
        if candidate in df.columns:
            label_col = candidate
            break

    # 展示 Label 分布（二分类）
    if label_col:
        print(f"\n--- 二分类标签 '{label_col}' ---")
        vc = df[label_col].value_counts().sort_index()
        total = len(df)
        for val, cnt in vc.items():
            print(f"  {str(val):>12s}: {cnt:>10,d}  ({cnt/total*100:6.2f}%)")
        print(f"  {'Total':>12s}: {total:>10,d}")

    # 展示 Attack 分布（多分类）
    if attack_col:
        print(f"\n--- 多分类标签 '{attack_col}' ---")
        vc = df[attack_col].value_counts().sort_values(ascending=False)
        total = len(df)
        for val, cnt in vc.items():
            print(f"  {str(val):>20s}: {cnt:>10,d}  ({cnt/total*100:6.2f}%)")
        print(f"  {'Total':>20s}: {total:>10,d}")

        # 小样本类别警告（<5000条）
        small_classes = vc[vc < 5000]
        if len(small_classes) > 0:
            print(f"\n以下类别样本 < 5000，考虑排除或特殊处理：")
            for val, cnt in small_classes.items():
                print(f"    {val}: {cnt:,d}")
    else:
        print("未找到攻击类别列，请手动检查列名")


def explore_missing_and_duplicates(df: pd.DataFrame):
    """3. 缺失值和重复记录"""
    section("3. 缺失值和重复记录检查")

    # 缺失值
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    print(f"总缺失值数: {total_nulls:,d}")

    if total_nulls > 0:
        cols_with_nulls = null_counts[null_counts > 0].sort_values(ascending=False)
        print("含缺失值的列：")
        for col, cnt in cols_with_nulls.items():
            print(f"  {col}: {cnt:,d} ({cnt/len(df)*100:.4f}%)")
    else:
        print("无缺失值")

    # Inf 检查（网络流量数据常见问题）
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_counts = np.isinf(df[numeric_cols]).sum()
    total_infs = inf_counts.sum()
    print(f"\n总 Inf 值数: {total_infs:,d}")

    if total_infs > 0:
        cols_with_infs = inf_counts[inf_counts > 0].sort_values(ascending=False)
        print("含 Inf 值的列：")
        for col, cnt in cols_with_infs.items():
            print(f"  {col}: {cnt:,d}")

    # 重复记录
    print(f"\n重复记录检查...")
    n_dup = df.duplicated().sum()
    print(f"完全重复的行: {n_dup:,d} ({n_dup/len(df)*100:.2f}%)")

    if n_dup > 0:
        # 检查去重后对类别分布的影响
        label_cols = [c for c in df.columns if c.lower() in ("label", "attack")]
        if label_cols:
            col = label_cols[-1]  # 优先用多分类标签
            print(f"\n去重前后 '{col}' 分布对比：")
            before = df[col].value_counts()
            after = df.drop_duplicates()[col].value_counts()
            comparison = pd.DataFrame({
                "Before": before,
                "After": after,
                "Dropped": before - after,
            }).fillna(0).astype(int)
            print(comparison.to_string())


def explore_statistics(df: pd.DataFrame):
    """4. 数值特征基本统计"""
    section("4. 数值特征统计摘要")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(f"数值特征数量: {len(numeric_cols)}")

    # 用 describe 生成统计，转置显示
    stats = df[numeric_cols].describe().T
    stats["null_pct"] = df[numeric_cols].isnull().mean() * 100

    # 格式化输出
    pd.set_option("display.max_rows", 50)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print(stats[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]].to_string())

    # 零方差特征检测
    zero_var = stats[stats["std"] == 0].index.tolist()
    if zero_var:
        print(f"\n零方差特征（可直接删除）: {zero_var}")

    # 高度偏斜特征
    skew = df[numeric_cols].skew()
    highly_skewed = skew[skew.abs() > 10].sort_values(ascending=False)
    if len(highly_skewed) > 0:
        print(f"\n高度偏斜特征 (|skewness| > 10):")
        for col, s in highly_skewed.items():
            print(f"  {col}: {s:.2f}")


def check_pipeline_compatibility(df: pd.DataFrame):
    """5. 与第一篇实验管道的兼容性检查"""
    section("5. 实验管道兼容性检查")

    # 5a. 确认全部为数值（NFv2应该已编码）
    non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
    label_candidates = [c for c in non_numeric if c.lower() in ("label", "attack")]
    truly_non_numeric = [c for c in non_numeric if c not in label_candidates]

    if truly_non_numeric:
        print(f"存在非数值特征列（需要编码）: {truly_non_numeric}")
    else:
        print("所有特征列均为数值类型（无需额外编码）")

    # 5b. Label 列检查
    for col in label_candidates:
        print(f"\n'{col}' 列值域: {sorted(df[col].unique())}")

    # 5c. IP 地址列检查（NFv2 通常包含 IPV4_SRC_ADDR 等）
    ip_cols = [c for c in df.columns if "ADDR" in c.upper() or "IP" in c.upper()]
    if ip_cols:
        print(f"\n检测到 IP 地址列: {ip_cols}")
        print("  这些列应从特征集中排除（标识符，非泛化特征）")
        for col in ip_cols:
            print(f"  {col} dtype: {df[col].dtype}, sample: {df[col].iloc[0]}")

    # 5d. 数据规模与内存估计
    mem_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
    print(f"\n数据集规模: {df.shape[0]:,d} 行 × {df.shape[1]} 列")
    print(f"内存占用: {mem_mb:.1f} MB")

    # 估算四个数据集总规模
    total_rows_estimate = 2.39e6 + 18.89e6 + 16.94e6 + 37.76e6
    scale_factor = total_rows_estimate / df.shape[0]
    print(f"四数据集总行数估计: ~{total_rows_estimate/1e6:.0f}M")
    print(f"相对当前数据集倍数: ~{scale_factor:.0f}x")
    print(f"建议: 对大数据集使用分块读取或采样策略")

    # 5e. NFv2 标准43特征列表对照
    # 来源: Sarhan et al. 2022 Table 3
    nfv2_expected_features = [
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
        "FTP_COMMAND_RET_CODE",
    ]
    # 注意: 上面是41个，NFv2论文中还有 IPV4_SRC_ADDR 和 IPV4_DST_ADDR
    # 共 43 列 + Label + Attack = 45 列

    print(f"\n--- NFv2 特征列名对照 ---")
    actual_cols = set(df.columns)
    expected_set = set(nfv2_expected_features)

    # 排除标签列后对照
    feature_actual = actual_cols - set(label_candidates)
    matched = feature_actual & expected_set
    missing = expected_set - feature_actual
    extra = feature_actual - expected_set

    print(f"匹配: {len(matched)}/{len(expected_set)}")
    if missing:
        print(f"预期但未出现: {sorted(missing)}")
    if extra:
        print(f"额外列（不在预期中）: {sorted(extra)}")


def main():
    args = parse_args()
    data_path = Path(args.data)

    if not data_path.exists():
        print(f"错误: 文件不存在 - {data_path}")
        sys.exit(1)

    print(f"正在读取: {data_path}")
    print(f"文件大小: {data_path.stat().st_size / 1024 / 1024:.1f} MB")

    # 读取数据
    df = pd.read_csv(data_path, low_memory=False)
    print(f"读取完成: {df.shape[0]:,d} 行 × {df.shape[1]} 列")

    # 运行全部探查
    explore_schema(df)
    explore_labels(df)
    explore_missing_and_duplicates(df)
    explore_statistics(df)
    check_pipeline_compatibility(df)

    section("探查完成")
    print("下一步：根据以上结果确认预处理策略，然后对其余三个数据集执行相同流程。\n")


if __name__ == "__main__":
    main()