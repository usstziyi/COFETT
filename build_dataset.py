"""
按 subject 遍历该被试预处理后的 FIF 文件。

读取的 FIF 由 preprocessing1.py 保存在：
    preprocess_output/<METHOD_STR>/<subject>/fif/*.fif

用法：
    uv run python build_dataset.py --subject sub-02
"""

import argparse
import re
from collections import Counter
from pathlib import Path
from typing import NamedTuple

import mne
import pandas as pd

ROOT_FOLDER = Path(__file__).resolve().parent
METHOD_STR = "prep"
PREP_ROOT = ROOT_FOLDER / "preprocess_output" / METHOD_STR
TEXT_DATASET = ROOT_FOLDER / "textdataset"

# 与 xlsx 行数做一致性检查的两个事件
READING_EVENT = "Reading start"
INNER_EVENT = "Inner-speech start"


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
    """把一个 fif 与其对应的 textdataset xlsx 配对，返回 (FifInfo, fif, xlsx)。"""
    info = parse_fif_name(fif_file)
    return info, fif_file, get_text_file(info.para, info.run)


def check_pair(fif_file, text_file):
    """检查 fif 事件数与配对 xlsx 行数是否一致。

    返回 (Reading 数, Inner-speech 数, xlsx 行数)。
    """
    raw = mne.io.read_raw_fif(fif_file, preload=False, verbose=False)
    counts = Counter(str(desc) for desc in raw.annotations.description)
    n_rows = len(pd.read_excel(text_file))
    return counts[READING_EVENT], counts[INNER_EVENT], n_rows


def main():
    parser = argparse.ArgumentParser(
        description="按 subject 遍历预处理后的 FIF 文件。"
    )
    parser.add_argument("--subject", default="sub-01", help="被试 ID（默认 sub-01）")
    args = parser.parse_args()

    fif_files = get_fif_files(args.subject)

    print(f"[Subject] {args.subject}")
    print(f"[FIF] {len(fif_files)} files")

    mismatches = 0
    for fif_file in fif_files:
        info, fif_file, text_file = pair_fif_with_text(fif_file)
        reading, inner, n_rows = check_pair(fif_file, text_file)

        ok = reading == n_rows and inner == n_rows
        if not ok:
            mismatches += 1
        flag = "OK" if ok else "MISMATCH"

        print(
            f"sub={info.sub} ses={info.ses} para={info.para} run={info.run} | "
            f"{text_file.name}: Reading={reading} Inner={inner} "
            f"rows={n_rows} -> {flag}"
        )

    print(f"[Check] {len(fif_files) - mismatches}/{len(fif_files)} matched")


if __name__ == "__main__":
    main()
