"""
读取指定 subject、指定若干 day 的全部桶数据，并按 n_chars 合并成组。

读取范围：
    preprocess_output/prep/<subject>/buckets/day-XX_bucket-size-NN.npz

用法：
    uv run python read_buckets.py --subject sub-02 --days 1 2 3 4
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


def load_days_buckets(subject, days):
    """加载某 subject 多天的桶，并把 n_chars 相同的样本合并成一组。

    同一 n_chars 在各天的桶具有相同的 n_points，所以可直接沿样本维拼接。
    返回按 n_chars 升序的 list[dict]，每项：
        n_chars / n_points / duration_s
        days              : 组内涉及的 day（升序）
        n_samples         : 组内样本总数
        n_samples_per_day : {day: 该天样本数}
        X                 : (n_samples, n_channels, n_points)
        y                 : list[str]，与 X 行序一致
        sample_days       : list[int]，与 X 行序一致的 day 标签
    """
    groups = {}
    for day in days:
        for X, y, meta in load_day_buckets(subject, day):
            group = groups.setdefault(
                meta["n_chars"],
                {
                    "n_chars": meta["n_chars"],
                    "n_points": meta["n_points"],
                    "duration_s": meta["duration_s"],
                    "n_samples_per_day": {},
                    "X": [],
                    "y": [],
                    "sample_days": [],
                },
            )
            group["X"].append(X)
            group["y"].extend(y)
            group["sample_days"].extend([meta["day"]] * meta["n_samples"])
            group["n_samples_per_day"][meta["day"]] = meta["n_samples"]

    result = []
    for n_chars in sorted(groups):
        group = groups[n_chars]
        group["X"] = np.concatenate(group["X"], axis=0)
        group["days"] = sorted(group["n_samples_per_day"])
        group["n_samples"] = len(group["y"])
        result.append(group)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="读取指定 subject、多个 day 的桶数据（按 n_chars 合并）。"
    )
    parser.add_argument("--subject", default="sub-01", help="被试 ID（默认 sub-01）")
    parser.add_argument(
        "--days", type=int, nargs="+", default=[1],
        help="天数 Day，可给多个（默认 1）",
    )
    args = parser.parse_args()

    groups = load_days_buckets(args.subject, args.days)

    print(f"[Subject] {args.subject}")
    print(f"[Days] {args.days}")
    print(f"[Groups] {len(groups)} 组（按 n_chars 合并）")

    n_total = 0
    for group in groups:
        n_total += group["n_samples"]
        print(
            f"  n_chars={group['n_chars']:>2} "
            f"n_points={group['n_points']:>5} "
            f"n_samples={group['n_samples']:>4} "
            f"X={group['X'].shape} {group['X'].dtype} "
            f"days={group['days']} per_day={group['n_samples_per_day']} "
            f"| {group['y'][0]}"
        )
    print(f"[Total] {n_total} samples")


if __name__ == "__main__":
    main()
