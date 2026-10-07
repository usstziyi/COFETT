"""
遍历某 subject 指定若干天的桶，把 y（句子文本）相同的样本归为一组。

读取范围：
    preprocess_output/prep/<subject>/buckets/day-XX_bucket-size-NN.npz
默认遍历第 5~8 天（para2）。

遍历顺序是 size（n_chars）优先：同一句话必然落在同一 size 的桶里，
因此可以逐 size 处理，每次只把该 size 在若干天的桶读进内存，处理完立即释放，
避免一次性把多天的全部桶都留在内存中。

归组结果一组一个 npz，存到
    preprocess_output/prep/<subject>/text_groups/text-XXXX.npz
并在同目录写 text_groups.csv 索引。

用法：
    uv run python group_by_text.py --subject sub-02
    uv run python group_by_text.py --subject sub-02 --days 5 6 7 8
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from read_buckets import (
    BUCKET_PATTERN,
    PREP_ROOT,
    list_bucket_files,
    load_bucket,
)

DEFAULT_DAYS = [5, 6, 7, 8]

ROOT_FOLDER = Path(__file__).resolve().parent
# days 5~8 对应 para2，使用 text2.xlsx
DEFAULT_TEXT_FILE = ROOT_FOLDER / "textdataset" / "text2.xlsx"


def list_buckets_by_size(subject, days):
    """返回 {n_chars: {day: 桶文件路径}}。

    只读文件名，不加载任何桶数据。
    """
    sizes = {}
    for day in days:
        for path in list_bucket_files(subject, day):
            n_chars = int(BUCKET_PATTERN.match(path.name).group(2))
            sizes.setdefault(n_chars, {})[day] = path
    return sizes


def group_by_text(subject, days):
    """把若干天的桶按句子文本归组。

    同一句子必然有相同的 n_chars / n_points，因此逐 size 处理时组内可直接堆叠。
    返回 list[dict]，按 (样本数降序, n_chars, 文本) 排序，每项：
        text              : 该组的句子
        n_chars / n_points: 句子对应的字数与点数
        days              : 组内样本涉及的 day（升序）
        n_samples         : 组内样本数（= 该句子被重复的次数）
        n_samples_per_day : {day: 该天样本数}
        X                 : (n_samples, n_channels, n_points)
        sample_days       : 与 X 行序一致的 day 标签
    """
    result = []

    for n_chars, day_paths in sorted(list_buckets_by_size(subject, days).items()):
        groups = {}

        for day in sorted(day_paths):
            X, y, meta = load_bucket(day_paths[day])
            for i, sentence in enumerate(y):
                group = groups.get(sentence)
                if group is None:
                    group = {
                        "text": sentence,
                        "n_chars": n_chars,
                        "n_points": meta["n_points"],
                        "X": [],
                        "sample_days": [],
                        "n_samples_per_day": {},
                    }
                    groups[sentence] = group

                group["X"].append(X[i])
                group["sample_days"].append(day)
                group["n_samples_per_day"][day] = (
                    group["n_samples_per_day"].get(day, 0) + 1
                )

        for group in groups.values():
            group["X"] = np.stack(group["X"])
            group["n_samples"] = len(group["sample_days"])
            group["days"] = sorted(group["n_samples_per_day"])
            result.append(group)

    result.sort(key=lambda g: (-g["n_samples"], g["n_chars"], g["text"]))
    return result


def check_text_coverage(groups, text_file=DEFAULT_TEXT_FILE):
    """检查 groups 的 text 与 text2.xlsx 的覆盖关系。

    text_file : xlsx 路径，默认 textdataset/text2.xlsx（days 5~8 对应 para2）

    返回 dict：
        text_file / n_rows / n_unique_text / n_group_text
        n_covered / coverage
        missing : xlsx 有、groups 没有的句子
        extra   : groups 有、xlsx 没有的句子
    """
    df = pd.read_excel(text_file)
    texts = {str(value) for value in df.iloc[:, 0]}
    group_texts = {group["text"] for group in groups}

    covered = texts & group_texts
    return {
        "text_file": Path(text_file).name,
        "n_rows": len(df),
        "n_unique_text": len(texts),
        "n_group_text": len(group_texts),
        "n_covered": len(covered),
        "coverage": len(covered) / len(texts) if texts else 1.0,
        "missing": sorted(texts - group_texts),
        "extra": sorted(group_texts - texts),
    }


def save_text_groups(groups, subject):
    """一组存成一个 npz，并在同目录写 text_groups.csv 索引。

    输出：preprocess_output/prep/<subject>/text_groups/
        text-0001.npz ... : 每组一个，内含 X / text / n_chars / n_points / sample_days
        text_groups.csv   : id 与句子、天数的对应关系

    因为分组 key 是中文句子（含标点、可能超长），文件名用编号而非 text。
    返回 (输出目录, 索引路径)。
    """
    out_dir = PREP_ROOT / subject / "text_groups"
    out_dir.mkdir(parents=True, exist_ok=True)

    index_rows = []
    for i, group in enumerate(groups, start=1):
        name = f"text-{i:04d}.npz"
        np.savez(
            out_dir / name,
            X=group["X"],
            text=np.array([group["text"]]),
            n_chars=group["n_chars"],
            n_points=group["n_points"],
            sample_days=np.array(group["sample_days"]),
        )
        index_rows.append(
            {
                "file": name,
                "text": group["text"],
                "n_chars": group["n_chars"],
                "n_points": group["n_points"],
                "n_samples": group["n_samples"],
                "days": ";".join(str(day) for day in group["days"]),
                "n_samples_per_day": ";".join(
                    f"{day}:{count}"
                    for day, count in sorted(group["n_samples_per_day"].items())
                ),
            }
        )

    index_path = out_dir / "text_groups.csv"
    pd.DataFrame(index_rows).to_csv(index_path, index=False, encoding="utf-8-sig")
    return out_dir, index_path


def main():
    parser = argparse.ArgumentParser(
        description="按句子文本把多天的桶样本归组。"
    )
    parser.add_argument("--subject", default="sub-01", help="被试 ID（默认 sub-01）")
    parser.add_argument(
        "--days", type=int, nargs="+", default=DEFAULT_DAYS,
        help=f"天数 Day，可给多个（默认 {' '.join(map(str, DEFAULT_DAYS))}）",
    )
    args = parser.parse_args()

    groups = group_by_text(args.subject, args.days)

    print(f"[Subject] {args.subject}")
    print(f"[Days] {args.days}")
    print(f"[Groups] {len(groups)} 组")

    # 重复次数分布：n_samples -> 组数
    counts = {}
    for group in groups:
        counts[group["n_samples"]] = counts.get(group["n_samples"], 0) + 1
    for n_samples in sorted(counts, reverse=True):
        print(f"  重复 {n_samples} 次的句子: {counts[n_samples]} 组")

    n_total = sum(group["n_samples"] for group in groups)
    print(f"[Total] {n_total} samples")

    print()
    print("[Preview] 前 5 组")
    for group in groups[:5]:
        print(
            f"  n={group['n_samples']} days={group['days']} "
            f"X={group['X'].shape} "
            f"| {group['text']}"
        )

    check = check_text_coverage(groups)
    print()
    print(
        f"[Text] {check['text_file']}: 行数 {check['n_rows']}，"
        f"去重 {check['n_unique_text']} 句"
    )
    print(
        f"[Coverage] {check['n_covered']}/{check['n_unique_text']} "
        f"({check['coverage'] * 100:.1f}%)，"
        f"groups 独有文本 {check['n_group_text'] - check['n_covered']} 句"
    )
    print(f"  missing (xlsx 有 / groups 无): {len(check['missing'])} 句")
    for text in check["missing"][:5]:
        print(f"    {text}")
    print(f"  extra (groups 有 / xlsx 无): {len(check['extra'])} 句")
    for text in check["extra"][:5]:
        print(f"    {text}")

    out_dir, index_path = save_text_groups(groups, args.subject)
    print()
    print(f"[Saved] {len(groups)} 组 -> {out_dir}")
    print(f"[Saved] 索引 -> {index_path.name}")


if __name__ == "__main__":
    main()
