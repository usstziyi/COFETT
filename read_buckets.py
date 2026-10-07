"""
读取指定 subject、指定 day 的全部桶数据。

读取范围：
    preprocess_output/prep/<subject>/buckets/day-XX_bucket-size-NN.npz

用法：
    uv run python read_buckets.py --subject sub-02 --day 1
"""

import argparse
import re
from pathlib import Path

import numpy as np

ROOT_FOLDER = Path(__file__).resolve().parent
PREP_ROOT = ROOT_FOLDER / "preprocess_output" / "prep"

# 桶文件名：day-XX_bucket-size-NN.npz
BUCKET_PATTERN = re.compile(r"^day-(\d+)_bucket-size-(\d+)\.npz$")


def get_bucket_dir(subject):
    """返回某被试的 buckets 目录。"""
    bucket_dir = PREP_ROOT / subject / "buckets"
    if not bucket_dir.exists():
        raise FileNotFoundError(f"buckets 目录不存在：{bucket_dir}")
    return bucket_dir


def list_bucket_files(subject, day):
    """列出某天的全部桶文件（按 n_chars 升序）。"""
    bucket_dir = get_bucket_dir(subject)

    files = []
    for path in bucket_dir.glob("*.npz"):
        match = BUCKET_PATTERN.match(path.name)
        if match is not None and int(match.group(1)) == day:
            files.append((int(match.group(2)), path))
    files.sort()

    if not files:
        raise FileNotFoundError(f"没有找到 day={day} 的桶：{bucket_dir}")
    return [path for _, path in files]


def load_bucket(path):
    """读取单个桶，返回 (X, y, meta)。

    X    : np.ndarray，形状 (n_samples, n_channels, n_points)
    y    : list[str]，与 X 一一对应的句子
    meta : dict，桶的元信息（由 npz 内的标量字段整理而来）
    """
    with np.load(path) as data:
        X = data["X"]
        y = [str(value) for value in data["y"]]
        meta = {
            "file": path.name,
            "day": int(data["day"]),
            "n_chars": int(data["n_chars"]),
            "n_points": int(data["n_points"]),
            "duration_s": float(data["duration_s"]),
            "onsets_s": data["onsets_s"].tolist(),
        }
    meta["n_samples"] = X.shape[0]
    return X, y, meta


def load_day_buckets(subject, day):
    """读取某 subject 某 day 的全部桶，返回 [(X, y, meta), ...]。"""
    return [load_bucket(path) for path in list_bucket_files(subject, day)]


def main():
    parser = argparse.ArgumentParser(
        description="读取指定 subject、day 的全部桶数据。"
    )
    parser.add_argument("--subject", default="sub-01", help="被试 ID（默认 sub-01）")
    parser.add_argument("--day", type=int, default=1, help="天数 Day（默认 1）")
    args = parser.parse_args()

    buckets = load_day_buckets(args.subject, args.day)

    print(f"[Subject] {args.subject}")
    print(f"[Day] {args.day}")
    print(f"[Buckets] {len(buckets)} 个")

    n_total = 0
    for X, y, meta in buckets:
        n_total += meta["n_samples"]
        print(
            f"  {meta['file']}: n_samples={meta['n_samples']} "
            f"X={X.shape} {X.dtype} "
            f"n_chars={meta['n_chars']} n_points={meta['n_points']} "
            f"| {y[0]}"
        )
    print(f"[Total] {n_total} samples")


if __name__ == "__main__":
    main()
