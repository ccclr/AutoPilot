#!/usr/bin/env python3
"""解析 logs 并将 SUMMARY 写入 results/bench-*.txt。"""

import argparse
import os
import sys

# 添加 benchmark 子目录到 Python 路径，以便导入 logs 模块
sys.path.append(os.path.join(os.path.dirname(__file__), "benchmark"))

from benchmark.logs import LogParser
from benchmark.utils import PathMaker


def _build_output_file(parser, faults):
    """按 bench 命名规则构造输出路径。"""
    nodes = int(parser.committee_size) if isinstance(parser.committee_size, int) else len(parser.rate)
    workers = int(parser.workers) if isinstance(parser.workers, int) else 1
    collocate = bool(parser.collocate)
    total_rate = int(sum(r for r in parser.rate if r is not None))
    tx_size = int(parser.size[0]) if parser.size else 0
    return PathMaker.result_file(faults, nodes, workers, collocate, total_rate, tx_size)


def analyze_logs(logs_dir, faults):
    """分析日志并将 SUMMARY 写入结果文件。"""
    parser = LogParser.process(logs_dir, faults=faults)
    summary_text = parser.result()
    preferred_output_file = _build_output_file(parser, faults)
    preferred_dir = os.path.dirname(preferred_output_file)
    os.makedirs(preferred_dir, exist_ok=True)

    output_file = preferred_output_file
    if os.path.exists(preferred_output_file) and not os.access(preferred_output_file, os.W_OK):
        # 目标文件不可写时，保留同命名规则并写到 analyzed 子目录。
        fallback_dir = os.path.join(preferred_dir, "analyzed")
        os.makedirs(fallback_dir, exist_ok=True)
        output_file = os.path.join(fallback_dir, os.path.basename(preferred_output_file))

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(summary_text)
    return output_file


def main():
    argp = argparse.ArgumentParser(description="将 logs 分析结果写入 results/bench-*.txt")
    argp.add_argument(
        "--logs-dir",
        default=os.path.join(os.path.dirname(__file__), "logs"),
        help="日志目录路径，默认 ./logs",
    )
    argp.add_argument(
        "--faults",
        type=int,
        default=0,
        help="故障节点数，默认 0",
    )
    args = argp.parse_args()

    logs_dir = os.path.abspath(args.logs_dir)
    if not os.path.isdir(logs_dir):
        print(f"错误：日志目录不存在 - {logs_dir}", file=sys.stderr)
        sys.exit(1)

    try:
        output_file = analyze_logs(logs_dir, faults=args.faults)
        print(f"已写入: {os.path.abspath(output_file)}")
    except Exception as e:
        print(f"错误：分析失败 - {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
