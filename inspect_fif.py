"""
读取预处理后的 FIF 文件信息（只读，不修改任何数据）。

读取的 FIF 由 preprocessing1.py 保存在：
    preprocess_output/<METHOD_STR>/<subject>/fif/*.fif

用法（只接收 fif 文件名，被试目录由文件名解析得到）：
    uv run python inspect_fif.py                                            # 遍历所有 fif，汇总保存为 CSV
    uv run python inspect_fif.py sub-02_ses-01_task-para1_run-01_eeg.fif    # 读取单个文件并打印
    uv run python inspect_fif.py D:/abs/path/xxx.fif                        # 也可直接给绝对路径

行为：
    - 指定 fif：把该文件的信息打印到终端
    - 未指定 fif：遍历所有被试的 fif，不打印，直接写入 CSV（见 FIF_SUMMARY_CSV）

信息字段：
    - 文件大小、时长
    - 通道总数与各类型通道数（EEG / EOG / STIM ...）
    - 采样率
    - 滤波设置（highpass / lowpass）
    - bad 通道
    - annotations 总数与各类事件数量（CSV 中每个事件类型单独一列）
"""

import argparse
from collections import Counter
from pathlib import Path

import mne
import pandas as pd

ROOT_FOLDER = Path(__file__).resolve().parent
METHOD_STR = "prep"
PREP_ROOT = ROOT_FOLDER / "preprocess_output" / METHOD_STR
FIF_SUMMARY_CSV = PREP_ROOT / "fif_summary.csv"


def parse_subject(fif_name):
    """从 BIDS 文件名解析被试目录名：sub-02_ses-01_..._eeg.fif -> sub-02。"""
    return Path(fif_name).name.split("_")[0]


def resolve_fif_file(fif_name):
    """把文件名解析成实际路径；被试目录由文件名解析得到。绝对路径直接使用。"""
    path = Path(fif_name)
    if path.is_absolute():
        return path
    return PREP_ROOT / parse_subject(fif_name) / "fif" / path.name


def collect_fif_files(fif_name):
    """fif_name 为空时扫描所有被试的 fif 目录，否则只读取该文件。"""
    if fif_name:
        path = resolve_fif_file(fif_name)
        if not path.exists():
            raise FileNotFoundError(f"FIF 文件不存在：{path}")
        return [path]

    files = sorted(PREP_ROOT.glob("*/fif/*.fif"))
    if not files:
        raise FileNotFoundError(f"没有找到 fif 文件：{PREP_ROOT}/*/fif/*.fif")
    return files


def read_fif_info(fif_file):
    """读取单个 fif 的元信息。

    使用 preload=False，MNE 只解析文件头，不会把整段数据读进内存。
    """
    raw = mne.io.read_raw_fif(fif_file, preload=False, verbose=False)

    ch_type_counter = Counter(raw.get_channel_types())
    sfreq = float(raw.info["sfreq"])
    duration = raw.n_times / sfreq if raw.n_times else 0.0
    event_counter = Counter(str(desc) for desc in raw.annotations.description)

    return {
        "file": fif_file.name,
        "size_mb": fif_file.stat().st_size / 1024**2,
        "n_channels": raw.info["nchan"],
        "ch_types": dict(ch_type_counter),
        "sfreq": sfreq,
        "highpass": float(raw.info["highpass"]),
        "lowpass": float(raw.info["lowpass"]),
        "duration_s": duration,
        "n_annotations": len(raw.annotations),
        "events": dict(event_counter),
        "bads": list(raw.info["bads"]),
    }


def format_ch_types(ch_types):
    return ", ".join(f"{ch_type}={count}" for ch_type, count in sorted(ch_types.items()))


def print_fif_info(info):
    print("-" * 80)
    print(f"文件: {info['file']}")
    print(f"  大小:      {info['size_mb']:.1f} MB")
    print(f"  通道总数:  {info['n_channels']}  ({format_ch_types(info['ch_types'])})")
    print(f"  采样率:    {info['sfreq']:.1f} Hz")
    print(
        f"  滤波:      highpass={info['highpass']:.2f} Hz, "
        f"lowpass={info['lowpass']:.2f} Hz"
    )
    print(
        f"  时长:      {info['duration_s']:.1f} s "
        f"({info['duration_s'] / 60.0:.2f} min)"
    )
    bads = info["bads"]
    print(f"  bad 通道:  {len(bads)}  {bads if bads else ''}")

    print(f"  事件总数:  {info['n_annotations']}")
    if info["events"]:
        print("  事件分布:")
        for desc, count in sorted(info["events"].items(), key=lambda item: (-item[1], item[0])):
            print(f"    {desc:<24} {count}")
    else:
        print("  事件分布:  (无)")


def build_rows(fif_files):
    """把多个 fif 的信息整理成 CSV 行，每个事件类型单独一列。"""
    rows = []
    for fif_file in fif_files:
        info = read_fif_info(fif_file)
        row = {
            "file": info["file"],
            "size_mb": round(info["size_mb"], 1),
            "n_channels": info["n_channels"],
            "ch_types": format_ch_types(info["ch_types"]),
            "sfreq": info["sfreq"],
            "highpass": info["highpass"],
            "lowpass": info["lowpass"],
            "duration_s": round(info["duration_s"], 1),
            "n_annotations": info["n_annotations"],
            "bads": ";".join(info["bads"]),
        }
        for desc, count in sorted(info["events"].items()):
            row[desc] = count
        rows.append(row)
    return rows


def save_summary_csv(fif_files):
    """遍历所有 fif，把汇总信息写入 CSV（不打印逐文件内容）。"""
    df = pd.DataFrame(build_rows(fif_files))
    FIF_SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(FIF_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    return FIF_SUMMARY_CSV


def main():
    parser = argparse.ArgumentParser(
        description="批量读取预处理后的 FIF 文件并打印基本信息（只读）。"
    )
    parser.add_argument(
        "--fif",
        nargs="?",
        default=None,
        help="fif 文件名，如 sub-02_ses-01_task-para1_run-01_eeg.fif；"
        "指定时打印该文件信息，省略时遍历所有 fif 并保存 CSV",
    )
    args = parser.parse_args()

    if args.fif:
        for fif_file in collect_fif_files(args.fif):
            print_fif_info(read_fif_info(fif_file))
        return

    csv_path = save_summary_csv(collect_fif_files(None))
    print(f"[Saved] {csv_path}")


if __name__ == "__main__":
    main()
