"""
按 subject 遍历该被试预处理后的 FIF 文件。

读取的 FIF 由 preprocessing1.py 保存在：
    preprocess_output/<METHOD_STR>/<subject>/fif/*.fif

用法：
    uv run python build_dataset.py --subject sub-02
"""

import argparse
import re
import unicodedata
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
REST_EVENT = "1.8 s rest start"

# 理论时长：阅读阶段 0.4s*字数，想象阶段 0.4s*(字数+1)
SEC_PER_CHAR = 0.4

# 正偏差超过该阈值（ms）的句子单独汇总
OVER_THRESHOLD_MS = 400

# 逐句偏差明细的列（负偏差与超阈值共用）
DETAIL_COLUMNS = [
    "phase", "fif", "index", "sentence",
    "n_chars", "actual_ms", "theory_ms", "diff_ms",
]


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


def collect_phase_onsets(raw):
    """按 onset 顺序收集三个阶段事件的 onset（annotations 本身已排序）。"""
    onsets = {"reading": [], "inner": [], "rest": []}
    for onset, desc in zip(raw.annotations.onset, raw.annotations.description):
        desc = str(desc)
        if desc == READING_EVENT:
            onsets["reading"].append(float(onset))
        elif desc == INNER_EVENT:
            onsets["inner"].append(float(onset))
        elif desc == REST_EVENT:
            onsets["rest"].append(float(onset))
    return onsets


def build_phase_rows(fif_file, text_file):
    """逐句计算阅读 / 想象阶段的实际时长与理论时长及偏差。"""
    raw = mne.io.read_raw_fif(fif_file, preload=False, verbose=False)
    onsets = collect_phase_onsets(raw)
    sentences = load_sentences(text_file)

    n = len(sentences)
    counts = {key: len(value) for key, value in onsets.items()}
    if not all(count == n for count in counts.values()):
        raise ValueError(
            f"事件数与句子数不一致：sentences={n}, "
            f"Reading={counts['reading']}, Inner={counts['inner']}, "
            f"Rest={counts['rest']}"
        )

    rows = []
    for i, sentence in enumerate(sentences):
        n_chars = count_chars(sentence)

        reading_onset = onsets["reading"][i]
        inner_onset = onsets["inner"][i]
        rest_onset = onsets["rest"][i]

        reading_actual_ms = (inner_onset - reading_onset) * 1000.0
        reading_theory_ms = SEC_PER_CHAR * 1000.0 * n_chars
        inner_actual_ms = (rest_onset - inner_onset) * 1000.0
        inner_theory_ms = SEC_PER_CHAR * 1000.0 * (n_chars + 1)

        rows.append(
            {
                "index": i,
                "sentence": sentence,
                "n_chars": n_chars,
                "reading_onset_s": round(reading_onset, 6),
                "inner_onset_s": round(inner_onset, 6),
                "rest_onset_s": round(rest_onset, 6),
                "reading_actual_ms": round(reading_actual_ms, 3),
                "reading_theory_ms": round(reading_theory_ms, 3),
                "reading_diff_ms": round(reading_actual_ms - reading_theory_ms, 3),
                "inner_actual_ms": round(inner_actual_ms, 3),
                "inner_theory_ms": round(inner_theory_ms, 3),
                "inner_diff_ms": round(inner_actual_ms - inner_theory_ms, 3),
            }
        )
    return rows


def save_phase_csv(fif_file, rows):
    """把逐句时长偏差写入 <subject>/analyse/<fif_stem>/phase_durations.csv。"""
    output = fif_file.parent.parent / "analyse" / fif_file.stem / "phase_durations.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False, encoding="utf-8-sig")
    return output


def summarize_diffs(values):
    """统计一组偏差：大于 / 小于理论值的个数及各自范围（ms）。"""
    positive = [v for v in values if v > 0]
    negative = [v for v in values if v < 0]
    return {
        "n_total": len(values),
        "n_gt": len(positive),
        "n_lt": len(negative),
        "n_eq": len(values) - len(positive) - len(negative),
        "gt_min_ms": round(min(positive), 3) if positive else None,
        "gt_max_ms": round(max(positive), 3) if positive else None,
        "lt_min_ms": round(min(negative), 3) if negative else None,
        "lt_max_ms": round(max(negative), 3) if negative else None,
    }


def _range_text(low, high):
    return "无" if low is None else f"{low:+.1f} ~ {high:+.1f} ms"


def detail_row(phase, fif_stem, row):
    """把一条偏差整理成明细行（phase 为 reading / inner）。"""
    return {
        "phase": phase,
        "fif": fif_stem,
        "index": row["index"],
        "sentence": row["sentence"],
        "n_chars": row["n_chars"],
        "actual_ms": row[f"{phase}_actual_ms"],
        "theory_ms": row[f"{phase}_theory_ms"],
        "diff_ms": row[f"{phase}_diff_ms"],
    }


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
    reading_diffs = []
    inner_diffs = []
    negative_rows = []
    over_rows = []
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

        rows = build_phase_rows(fif_file, text_file)
        output = save_phase_csv(fif_file, rows)

        reading_diffs.extend(r["reading_diff_ms"] for r in rows)
        inner_diffs.extend(r["inner_diff_ms"] for r in rows)

        for r in rows:
            if r["reading_diff_ms"] < 0:
                negative_rows.append(detail_row("reading", fif_file.stem, r))
            elif r["reading_diff_ms"] > OVER_THRESHOLD_MS:
                over_rows.append(detail_row("reading", fif_file.stem, r))

            if r["inner_diff_ms"] < 0:
                negative_rows.append(detail_row("inner", fif_file.stem, r))
            elif r["inner_diff_ms"] > OVER_THRESHOLD_MS:
                over_rows.append(detail_row("inner", fif_file.stem, r))

        print(f"    [Saved] {output}")

    print(f"[Check] {len(fif_files) - mismatches}/{len(fif_files)} matched")

    # --------------------------------------------------------
    # 全局汇总：大于 / 小于理论值的个数与范围
    # --------------------------------------------------------
    summary_rows = []
    for phase, values in [("reading", reading_diffs), ("inner", inner_diffs)]:
        summary_rows.append({"phase": phase, **summarize_diffs(values)})

    summary_df = pd.DataFrame(
        summary_rows,
        columns=[
            "phase", "n_total", "n_gt", "n_lt", "n_eq",
            "gt_min_ms", "gt_max_ms", "lt_min_ms", "lt_max_ms",
        ],
    )
    summary_path = PREP_ROOT / args.subject / "analyse" / "phase_durations_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print()
    print("=" * 70)
    print(f"[Summary] 总句子数 {len(reading_diffs)}")
    for row in summary_rows:
        print(
            f"  {row['phase']:<8} "
            f"大于理论值 {row['n_gt']:>5} 个 "
            f"(范围 {_range_text(row['gt_min_ms'], row['gt_max_ms'])}) | "
            f"小于理论值 {row['n_lt']:>5} 个 "
            f"(范围 {_range_text(row['lt_min_ms'], row['lt_max_ms'])}) | "
            f"等于 {row['n_eq']}"
        )
    print(f"[Saved] {summary_path}")

    # --------------------------------------------------------
    # 负偏差明细：单独一个 CSV，每行一条（含句子文本）
    # --------------------------------------------------------
    negative_path = summary_path.parent / "phase_durations_negative.csv"
    pd.DataFrame(negative_rows, columns=DETAIL_COLUMNS).to_csv(
        negative_path, index=False, encoding="utf-8-sig"
    )
    print(f"[Saved] {negative_path} ({len(negative_rows)} rows)")

    # --------------------------------------------------------
    # 正偏差超过 OVER_THRESHOLD_MS 的明细：单独一个 CSV
    # --------------------------------------------------------
    over_path = summary_path.parent / f"phase_durations_over{OVER_THRESHOLD_MS}ms.csv"
    pd.DataFrame(over_rows, columns=DETAIL_COLUMNS).to_csv(
        over_path, index=False, encoding="utf-8-sig"
    )
    print(f"[Saved] {over_path} ({len(over_rows)} rows)")


if __name__ == "__main__":
    main()
