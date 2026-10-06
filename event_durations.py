"""
统计单个 FIF 文件中所有事件的持续时长（毫秒）。

背景：
preprocessing1.py 写入的 annotations 都是“点事件”，
duration 全为 0，因此单个事件本身没有时长可读。
这里把“事件持续时长”定义为：
    当前事件 onset -> 下一个事件 onset 的时间间隔。
例如：
    Reading start      -> Inner-speech start    = 阅读阶段时长
    Inner-speech start -> 1.8 s rest start      = 内语阶段时长
    1.8 s rest start   -> 下一个 Reading start  = 1.8 s 休息时长
最后一个事件没有“下一个事件”，其持续时长记为空。

用法：
    uv run python event_durations.py <fif文件名>
    uv run python event_durations.py sub-01_ses-01_task-para1_run-01_eeg.fif

<fif文件名> 会拼接在 FIF_FOLDER（默认 preprocess_output/prep/sub-01/fif）之下；
也可以直接传绝对路径，此时 FIF_FOLDER 会被忽略。

结果写入 analyse/<fif文件名>/ 目录（analyse 与 fif 同级），
每个事件类别一个独立 CSV（类内按时间顺序），例如：
    analyse/sub-01_ses-01_task-para1_run-01_eeg/Reading_start.csv
    analyse/sub-01_ses-01_task-para1_run-01_eeg/Inner-speech_start.csv
    analyse/sub-01_ses-01_task-para1_run-01_eeg/1.8_s_rest_start.csv
"""

import argparse
import csv
import re
from pathlib import Path

import mne

ROOT_FOLDER = Path(__file__).resolve().parent
FIF_FOLDER = (
    ROOT_FOLDER
    / "preprocess_output"
    / "prep"
    / "sub-01"
    / "fif"
)
# 分析结果目录，与 fif 同级；每个 fif 再各自建一个子目录
ANALYSE_FOLDER = FIF_FOLDER.parent / "analyse"


def event_durations(fif_file):
    """读取 FIF 中的 annotations，返回逐事件的持续时长行。"""
    raw = mne.io.read_raw_fif(fif_file, preload=False, verbose=False)
    annotations = raw.annotations

    onsets = [float(value) for value in annotations.onset]
    descriptions = [str(value) for value in annotations.description]

    if not onsets:
        raise ValueError(f"{fif_file} 中没有任何 annotations。")

    # MNE 的 Annotations 在构造和 append 时都会调用 _sort()，
    # 从 FIF 读出的 onset 必然按升序排列，无需再排序。

    rows = []
    for i, (onset, description) in enumerate(zip(onsets, descriptions)):
        if i + 1 < len(onsets):
            duration_ms = round((onsets[i + 1] - onset) * 1000.0, 3)
            next_description = descriptions[i + 1]
        else:
            duration_ms = ""
            next_description = ""
        rows.append(
            {
                "index": i,
                "event": description,
                "onset_s": round(onset, 6),
                "duration_ms": duration_ms,
                "next_event": next_description,
            }
        )

    return rows


def safe_filename(event):
    """把事件描述转成安全的文件名（空格等非字母数字字符替换为下划线）。"""
    return re.sub(r"[^0-9A-Za-z._-]+", "_", event).strip("_")


def save_category_csvs(rows, output_dir):
    """把事件按类别分别写入 output_dir 下的独立 CSV，类内按时间顺序。

    返回 [(事件描述, 文件路径, 该类事件数), ...]，顺序为各类别首次出现的时间顺序。
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # dict 保持插入顺序，即各类别第一次出现的时间顺序
    groups = {}
    for row in rows:
        groups.setdefault(row["event"], []).append(row)

    fieldnames = ["index", "event", "onset_s", "duration_ms", "next_event"]
    written = []
    for event, event_rows in groups.items():
        path = output_dir / f"{safe_filename(event)}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for local_index, row in enumerate(event_rows):
                writer.writerow(
                    {
                        "index": local_index,
                        "event": row["event"],
                        "onset_s": row["onset_s"],
                        "duration_ms": row["duration_ms"],
                        "next_event": row["next_event"],
                    }
                )
        written.append((event, path, len(event_rows)))

    return written


def main():
    parser = argparse.ArgumentParser(
        description="统计单个 FIF 文件中所有事件的持续时长（ms）。"
    )
    parser.add_argument("fif", help="fif 目录下的 FIF 文件名（或绝对路径）")
    args = parser.parse_args()

    fif_file = Path(FIF_FOLDER) / args.fif
    rows = event_durations(fif_file)

    output_dir = ANALYSE_FOLDER / fif_file.stem
    written = save_category_csvs(rows, output_dir)

    print(f"[Events] {len(rows)}")
    for event, path, count in written:
        print(f"[Saved] {event}: {count} -> {path}")


if __name__ == "__main__":
    main()
