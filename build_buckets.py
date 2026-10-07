"""
按 subject 遍历该被试预处理后的 FIF 文件，基于 Inner-speech 事件切分数据集。

规则：
    X：事件起点 = "Inner-speech start"
       事件时长 = (text 中非标点字符数 n + 1) * 0.4 s
       截取点数 = (n + 1) * 0.4 * 500（严格按此公式计算）
       取 EEG 通道数据，形状 (n_channels, n_points)
    y：事件对应的 text

按 (day, n_points) 分桶后，每个桶存成
    preprocess_output/prep/<subject>/buckets/day-XX_bucket-size-NN.npz
并在同目录下写一份 buckets.csv 汇总每个桶的信息。
（格式：npz，内含 X / y 及事件元信息）。

用法：
    uv run python build_buckets.py --subject sub-02
"""

import argparse
import re
import unicodedata
from pathlib import Path
from typing import NamedTuple

import mne
import numpy as np
import pandas as pd

# ============================================================
# 路径配置（原 build_dataset.py，本地化一份便于改动）
# ============================================================
ROOT_FOLDER = Path(__file__).resolve().parent
METHOD_STR = "prep"
PREP_ROOT = ROOT_FOLDER / "preprocess_output" / METHOD_STR
TEXT_DATASET = ROOT_FOLDER / "textdataset"

# 事件描述
INNER_EVENT = "Inner-speech start"

# 单个字符对应的事件时长
SEC_PER_CHAR = 0.4

# 与 preprocessing1.py 中 SAMPLE_RATE 保持一致
SFREQ = 500


# ============================================================
# 以下为原 build_dataset.py 中复制过来的辅助函数
# ============================================================
def get_fif_files(subject):
    """返回某被试 fif 目录下的全部 fif 文件（按名称排序）。"""
    fif_dir = PREP_ROOT / subject / "fif"
    if not fif_dir.exists():
        raise FileNotFoundError(f"FIF 目录不存在：{fif_dir}")

    files = sorted(fif_dir.glob("*.fif"))
    if not files:
        raise FileNotFoundError(f"目录下没有 fif 文件：{fif_dir}")
    return files


class FifInfo(NamedTuple):
    """从 fif 文件名解析出的 BIDS 字段，全部为 int。"""

    sub: int
    ses: int
    para: int
    run: int


def _extract_int(pattern, name):
    match = re.search(pattern, name)
    if match is None:
        raise ValueError(f"无法从文件名解析 {pattern!r}：{name}")
    return int(match.group(1))


def parse_fif_name(fif_name):
    """从 BIDS 文件名解析 sub / ses / task-para / run（去前导 0，全部 int）。

    例：sub-02_ses-01_task-para1_run-01_eeg.fif -> FifInfo(2, 1, 1, 1)
    """
    name = Path(fif_name).name
    return FifInfo(
        sub=_extract_int(r"sub-(\d+)", name),
        ses=_extract_int(r"ses-(\d+)", name),
        para=_extract_int(r"para(\d+)", name),
        run=_extract_int(r"run-(\d+)", name),
    )


def get_text_file(para, run):
    """根据 para / run 返回 textdataset 下对应的 xlsx 路径。

    配对规则：
        para1 run1..4 -> text1-1 .. text1-4.xlsx
        para2 任意 run -> text2.xlsx
    """
    if para == 1:
        name = f"text1-{run}.xlsx"
    elif para == 2:
        name = "text2.xlsx"
    else:
        raise ValueError(f"未知 para：{para}")

    path = TEXT_DATASET / name
    if not path.exists():
        raise FileNotFoundError(f"找不到对应的文本文件：{path}")
    return path


def pair_fif_with_text(fif_file):
    """把一个 fif 与其对应的 textdataset xlsx 配对，返回 (FifInfo, xlsx)。"""
    info = parse_fif_name(fif_file)
    return info, get_text_file(info.para, info.run)


def get_day(info):
    """由 ses / para 推算采集天数（Day 从 1 开始）。

    para1 的 ses1..4 为 Day1~Day4，para2 的 ses1..4 为 Day5~Day8。
    """
    return (info.para - 1) * 4 + info.ses


def load_sentences(text_file):
    """读取 xlsx 的第一列，返回句子列表（与事件一一对应）。"""
    df = pd.read_excel(text_file)
    return [str(value) for value in df.iloc[:, 0]]


def count_chars(sentence):
    """统计字数：标点符号与空白不计入。

    用 Unicode 类别判断，标点（P*）和分隔符/空白（Z*）都跳过，
    汉字、字母、数字正常计数。
    """
    return sum(
        1
        for ch in sentence
        if not unicodedata.category(ch).startswith(("P", "Z"))
    )


def get_inner_onsets(raw):
    """按 onset 顺序取出所有 Inner-speech 事件的 onset。"""
    return [
        float(onset)
        for onset, desc in zip(raw.annotations.onset, raw.annotations.description)
        if str(desc) == INNER_EVENT
    ]


# ============================================================
# 数据集切分
# ============================================================
def extract_inner_events(fif_file, text_file, day):
    """从一个 fif + text 中切出所有 Inner-speech 事件。

    返回 (X, y, rows)：
        X    : list[np.ndarray]，每个元素形状 (n_channels, n_points)
        y    : list[str]，与 X 一一对应的句子
        rows : list[dict]，每个事件的元信息（含采集天 day，用于打印校验）
    """
    # 不整段 preload，避免长记录占用过多内存；
    # 只按需读取每个事件所在的时间窗。
    raw = mne.io.read_raw_fif(fif_file, preload=False, verbose=False)

    if int(round(float(raw.info["sfreq"]))) != SFREQ:
        raise ValueError(
            f"采样率 {raw.info['sfreq']} Hz 与预期 {SFREQ} Hz 不一致"
        )

    inner_onsets = get_inner_onsets(raw)
    sentences = load_sentences(text_file)

    if len(inner_onsets) != len(sentences):
        raise ValueError(
            f"Inner 事件数 {len(inner_onsets)} 与文本行数 {len(sentences)} 不一致"
        )

    n_channels = len(mne.pick_types(raw.info, eeg=True))
    n_total = raw.n_times

    X, y, rows = [], [], []
    for i, sentence in enumerate(sentences):
        n_chars = count_chars(sentence)

        # 严格按公式计算点数：(n + 1) * 0.4 * 500
        n_points = int(round((n_chars + 1) * SEC_PER_CHAR * SFREQ))

        start = int(round(inner_onsets[i] * SFREQ))
        end = start + n_points

        if end <= n_total:
            segment = raw.get_data(picks="eeg", start=start, stop=end)
            padded = 0
        else:
            # 超出记录末尾时补零，保证点数严格等于 n_points
            segment = np.zeros((n_channels, n_points), dtype=float)
            keep = max(n_total - start, 0)
            if keep > 0:
                segment[:, :keep] = raw.get_data(
                    picks="eeg", start=start, stop=n_total
                )
            padded = n_points - keep

        X.append(segment)
        y.append(sentence)
        rows.append(
            {
                "day": day,
                "index": i,
                "sentence": sentence,
                "n_chars": n_chars,
                "onset_s": round(float(inner_onsets[i]), 6),
                "n_points": n_points,
                "padded": padded,
            }
        )

    return X, y, rows


def build_buckets(X, y, rows):
    """按 (day, n_points) 分桶：同一天内 n_points 相同的样本进同一个桶。

    rows 为全量事件元信息（顺序与 X / y 一致），
    返回按 (day, n_points) 排序的桶列表，每个桶为一个描述 dict：
        day / n_points / n_chars / duration_s / n_samples / n_padded
        idx / onsets_s / X / y
    其中：
        idx : 桶内样本在全局 X、y 中的下标
        X   : 桶内样本堆叠结果，形状 (n_samples, n_channels, n_points)
        y   : 桶内样本对应的句子
    """
    buckets = {}
    for idx, row in enumerate(rows):
        key = (row["day"], row["n_points"])
        bucket = buckets.get(key)
        if bucket is None:
            bucket = {
                "day": row["day"],
                "n_points": row["n_points"],
                "n_chars": row["n_chars"],
                "duration_s": round(row["n_points"] / SFREQ, 3),
                "idx": [],
                "onsets_s": [],
                "n_padded": 0,
            }
            buckets[key] = bucket

        bucket["idx"].append(idx)
        bucket["onsets_s"].append(row["onset_s"])
        bucket["n_padded"] += row["padded"] > 0

    result = sorted(buckets.values(), key=lambda b: (b["day"], b["n_points"]))
    for bucket in result:
        bucket["n_samples"] = len(bucket["idx"])
        bucket["X"] = np.stack([X[i] for i in bucket["idx"]])
        bucket["y"] = [y[i] for i in bucket["idx"]]
    return result


def get_bucket_path(subject, bucket):
    """返回某个桶的保存路径：<subject>/buckets/day-XX_bucket-size-NN.npz。"""
    return (
        PREP_ROOT / subject / "buckets"
        / f"day-{bucket['day']:02d}_bucket-size-{bucket['n_chars']:02d}.npz"
    )


def save_buckets(buckets, subject):
    """把每个桶存成一个 npz（含 X / y 及事件元信息），返回保存目录与路径列表。"""
    out_dir = PREP_ROOT / subject / "buckets"
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = []
    for bucket in buckets:
        path = get_bucket_path(subject, bucket)
        np.savez(
            path,
            X=bucket["X"],
            y=np.array(bucket["y"]),
            day=bucket["day"],
            n_chars=bucket["n_chars"],
            n_points=bucket["n_points"],
            duration_s=bucket["duration_s"],
            onsets_s=np.array(bucket["onsets_s"]),
        )
        paths.append(path)
    return out_dir, paths


def save_bucket_index(buckets, subject):
    """在 <subject>/buckets/buckets.csv 列出每个桶的信息（不含 EEG 数据）。"""
    out_dir = PREP_ROOT / subject / "buckets"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "file": get_bucket_path(subject, bucket).name,
            "day": bucket["day"],
            "n_chars": bucket["n_chars"],
            "n_points": bucket["n_points"],
            "duration_s": bucket["duration_s"],
            "n_samples": bucket["n_samples"],
            "n_padded": bucket["n_padded"],
        }
        for bucket in buckets
    ]

    path = out_dir / "buckets.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def main():
    parser = argparse.ArgumentParser(
        description="按 subject 切分 Inner-speech EEG 数据集（不存盘）。"
    )
    parser.add_argument("--subject", default="sub-01", help="被试 ID（默认 sub-01）")
    args = parser.parse_args()

    fif_files = get_fif_files(args.subject)

    print(f"[Subject] {args.subject}")
    print(f"[FIF] {len(fif_files)} files")

    all_X, all_y, all_rows = [], [], []
    n_padded_total = 0

    for fif_file in fif_files:
        info, text_file = pair_fif_with_text(fif_file)
        day = get_day(info)
        try:
            X, y, rows = extract_inner_events(fif_file, text_file, day)
        except Exception as exc:
            print(f"[Skip] {fif_file.name}: {exc}")
            continue

        all_X.extend(X)
        all_y.extend(y)
        all_rows.extend(rows)

        n_padded = sum(r["padded"] > 0 for r in rows)
        n_padded_total += n_padded
        lengths = [r["n_points"] for r in rows]

        print(
            f"sub={info.sub} ses={info.ses} para={info.para} run={info.run} "
            f"day={day} | "
            f"{text_file.name}: events={len(X)} "
            f"channels={X[0].shape[0]} "
            f"points {min(lengths)}~{max(lengths)} "
            f"padded={n_padded}"
        )

    print()
    print("=" * 70)
    print(f"[Dataset] X = {len(all_X)} events, y = {len(all_y)} sentences")
    if all_X:
        print(f"  channels = {all_X[0].shape[0]}")
        print(f"  points/channels 示例 = {all_X[0].shape[1]}")
    print(f"[Padded] {n_padded_total} events 因超出记录末尾补零")

    buckets = build_buckets(all_X, all_y, all_rows)
    print()
    print(f"[Buckets] 共 {len(buckets)} 个桶（key = day + n_points）")
    current_day = None
    for bucket in buckets:
        if bucket["day"] != current_day:
            current_day = bucket["day"]
            day_buckets = [b for b in buckets if b["day"] == current_day]
            print(
                f"  Day{current_day}: {len(day_buckets)} 桶, "
                f"{sum(b['n_samples'] for b in day_buckets)} samples"
            )
        print(
            f"    n_points={bucket['n_points']:>5} "
            f"n_chars={bucket['n_chars']:>2} "
            f"{bucket['duration_s']:>5.1f}s "
            f"n={bucket['n_samples']:>3}"
        )

    out_dir, paths = save_buckets(buckets, args.subject)
    index_path = save_bucket_index(buckets, args.subject)
    print()
    print(f"[Saved] {len(paths)} 个桶 -> {out_dir}")
    print(f"[Saved] 桶信息索引 -> {index_path.name}")

    if all_rows:
        print()
        print("[Preview] 前 5 个事件")
        for r in all_rows[:5]:
            print(
                f"  day={r['day']} #{r['index']}: n_chars={r['n_chars']} "
                f"onset={r['onset_s']}s points={r['n_points']} | {r['sentence']}"
            )


if __name__ == "__main__":
    main()
