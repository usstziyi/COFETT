import os
import re
from pathlib import Path
from collections import Counter

import mne
import numpy as np
import pandas as pd

from pyprep.prep_pipeline import PrepPipeline
from mne_icalabel import label_components


# ============================================================
# 1. Configuration
# ============================================================
ROOT_FOLDER = Path(__file__).resolve().parent
SUBJECT_ID = "sub-01"
METHOD_STR = "prep"
SAMPLE_RATE = 500

TEST = False
PREP = True
ICA = True


# 是否在 PREP 中启用 RANSAC
# RANSAC 更慢，但可以利用电极空间位置检测异常通道


# Chisco-2.0 中这两个通道仍被标为 EEG，
# 但旧 Chisco montage.csv 中坐标为 (0, 0, 0)
INVALID_EEG_CHANNELS = ["10", "111"]

# 辅助通道
EOG_CHANNELS = ["VEO", "HEO"]
STIM_CHANNEL = "Trigger"

# Chisco-2.0 事件编码 -> annotations 描述
EVENT_DESCRIPTIONS = {
    65329: "Reading start",
    65379: "Inner-speech start",
    65381: "1.8 s rest start",
    65480: "long rest start",
    65481: "long rest end",
}

# 原 Chisco montage 中不应该作为 Chisco-2.0 scalp EEG 使用的条目
MONTAGE_EXCLUDE_CHANNELS = {
    "10", "11", "111", "110", "84", "85",
    "EKG", "EMG", "VEO", "HEO", "Trigger",
}

MONTAGE_FILE = ROOT_FOLDER / "montage.csv"
TEXTDATASET = ROOT_FOLDER / "textdataset"


# ============================================================
# 2. Dataset path
# ============================================================
if SUBJECT_ID == "sub-01":
    DATA_FOLDER = ROOT_FOLDER / "Chisco-2.0" / "sub-01"
elif SUBJECT_ID == "sub-02":
    DATA_FOLDER = ROOT_FOLDER / "Chisco-2.0" / "sub-02"
else:
    raise ValueError(f"Invalid subject id: {SUBJECT_ID}")


# ============================================================
# 3. Output directories
# ============================================================
OUTPUT_FOLDER = ROOT_FOLDER / "preprocess_output" / f"{METHOD_STR}/{SUBJECT_ID}"
FIF_DIR = OUTPUT_FOLDER / "fif"
QC_DIR = OUTPUT_FOLDER / "qc"

for folder in [OUTPUT_FOLDER, FIF_DIR, QC_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


# ============================================================
# 4. Montage
# ============================================================
def load_chisco2_montage(montage_file: Path):
    """
    从 Chisco 1.0 montage.csv 构建 Chisco-2.0 montage。

    只保留真正有空间坐标的 122 个 scalp EEG 通道。

    注意：
    这里没有人为给 10 / 111 猜坐标，而是直接删除。
    """
    df = pd.read_csv(montage_file)

    required_columns = {"label", "x", "y", "z"}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Montage must contain columns: {required_columns}")

    # 去掉空行
    df = df.dropna(subset=["label", "x", "y", "z"]).copy()
    df["label"] = df["label"].astype(str)

    # 删除 Chisco-2.0 不使用的空间通道
    df = df[~df["label"].isin(MONTAGE_EXCLUDE_CHANNELS)].copy()

    # 转换坐标
    for column in ["x", "y", "z"]:
        df[column] = pd.to_numeric(df[column], errors="raise")

    # 检查是否还有 (0, 0, 0)
    zero_mask = np.isclose(df[["x", "y", "z"]].to_numpy(), 0.0).all(axis=1)
    if zero_mask.any():
        zero_channels = df.loc[zero_mask, "label"].tolist()
        raise ValueError(
            "Montage still contains zero-position EEG channels: "
            f"{zero_channels}"
        )

    if len(df) != 122:
        raise ValueError(
            f"Expected 122 EEG channels in montage, but got {len(df)}."
        )

    ch_pos = {
        row["label"]: np.array([row["x"], row["y"], row["z"]], dtype=float)
        for _, row in df.iterrows()
    }

    # 保留原 montage 的坐标语义。
    # 因为旧文件没有 nasion/LPA/RPA，
    # MNE 仍可能提示 fiducial warning，这是非致命的。
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="unknown")

    return montage


MONTAGE = load_chisco2_montage(MONTAGE_FILE)
print(f"[Montage] {len(MONTAGE.ch_names)} EEG positions loaded.")


# ============================================================
# 5. EDF / events pairing
# ============================================================
def get_bids_prefix(path: Path, suffix: str):
    """
    例如：

    sub-01_ses-01_task-para1_run-01_eeg.edf

    ->
    sub-01_ses-01_task-para1_run-01
    """
    name = path.name
    marker = f"_{suffix}"

    if marker not in name:
        raise ValueError(f"Cannot extract prefix from {path.name}")

    return name.split(marker)[0]


def pair_edf_and_events(eeg_dir: Path):
    """
    根据 BIDS 前缀严格匹配 EDF 和 events.tsv。

    不直接 zip(sorted(...))，
    防止少一个文件后所有文件发生错位。
    """
    edf_files = sorted(eeg_dir.glob("*_eeg.edf"))
    events_files = sorted(eeg_dir.glob("*_events.tsv"))

    edf_map = {get_bids_prefix(path, "eeg"): path for path in edf_files}
    events_map = {get_bids_prefix(path, "events"): path for path in events_files}

    edf_keys = set(edf_map)
    event_keys = set(events_map)

    missing_events = sorted(edf_keys - event_keys)
    missing_edf = sorted(event_keys - edf_keys)

    if missing_events:
        raise RuntimeError(
            "EDF without matching events.tsv:\n" + "\n".join(missing_events)
        )

    if missing_edf:
        raise RuntimeError(
            "events.tsv without matching EDF:\n" + "\n".join(missing_edf)
        )

    prefixes = sorted(edf_keys)

    return [(edf_map[prefix], events_map[prefix]) for prefix in prefixes]


# ============================================================
# 6. Events check
# ============================================================
def load_events_file(events_file: Path):
    """
    加载 Chisco-2.0 events.tsv。

    onset 列是相对记录起点的秒数（对应原始 1000 Hz 记录），
    sample 列对应原始 1000 Hz 采样点，这里都原样保留、不做换算。

    后续直接用 onset（秒）构建 annotations：
    onset 与采样率无关，重采样不需要任何手工换算。
    """
    events_df = pd.read_csv(events_file, sep="\t")

    required_columns = {"onset", "trial_type", "value", "sample"}
    missing = required_columns - set(events_df.columns)
    if missing:
        raise ValueError(f"{events_file.name} is missing columns: {missing}")

    events_df["onset"] = pd.to_numeric(events_df["onset"], errors="raise")
    events_df["value"] = pd.to_numeric(events_df["value"], errors="raise")
    events_df["sample"] = pd.to_numeric(events_df["sample"], errors="raise")

    return events_df


def build_annotations(events_df):
    """
    用 events.tsv 的 onset（秒，对应原始 1000 Hz 记录）构建 mne.Annotations。

    onset 的单位是秒，与采样率无关：
    先 set_annotations，再 raw.resample 时，
    MNE 只改变数据点密度，annotations 的 onset 保持不变，
    因此事件时刻不会因为重采样而漂移。

    事件编码通过 EVENT_DESCRIPTIONS 映射为可读描述；
    若出现未登记的事件编码则直接报错，避免静默丢事件。
    """
    descriptions = events_df["value"].map(EVENT_DESCRIPTIONS)

    unknown = sorted(events_df.loc[descriptions.isna(), "value"].unique())
    if unknown:
        raise ValueError(
            f"Event codes without description mapping: {unknown}. "
            "Please add them to EVENT_DESCRIPTIONS."
        )

    return mne.Annotations(
        onset=events_df["onset"].to_numpy(dtype=float),
        duration=np.zeros(len(events_df), dtype=float),
        description=descriptions.tolist(),
    )


# ============================================================
# 7. Helpers
# ============================================================
def to_str_list(values):
    """
    PyPREP 某些情况下可能返回 numpy.str_，
    统一转为 Python str。
    """
    if values is None:
        return []

    return [str(value) for value in values]


def get_eeg_channels(raw):
    """
    显式返回所有 EEG channel name。

    这里不依赖 pick_types 默认的 bad channel 排除行为。
    """
    return [
        ch_name
        for ch_name, ch_type in zip(raw.ch_names, raw.get_channel_types())
        if ch_type == "eeg"
    ]


# ============================================================
# 8. Process one EDF
# ============================================================
def process_edf_file(edf_file: Path, events_file: Path):
    print()
    print("=" * 80)
    print(f"Processing: {edf_file.name}")
    print("=" * 80)

    # --------------------------------------------------------
    # 8.1 Read raw EDF
    # --------------------------------------------------------
    raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
    print(f"[Raw] channels = {raw.info['nchan']}")
    print(f"[Raw] sfreq = {raw.info['sfreq']} Hz")

    # Chisco-2.0 原始应该是 127
    if raw.info["nchan"] != 127:
        print(f"[WARNING] Expected 127 raw channels, got {raw.info['nchan']}.")

    # --------------------------------------------------------
    # 8.2 Load events -> annotations, then resample
    # --------------------------------------------------------
    # annotations 的 onset 单位是秒，与采样率无关。
    # 先用原始 1000 Hz 记录的 onset 建立 annotations，
    # 再重采样：MNE 只改变数据点密度，onset 保持不变，
    # 因此事件时刻不会漂移，annotations 也会随 raw 一起保存。
    events_df = load_events_file(events_file)
    raw.set_annotations(build_annotations(events_df))

    event_counts = events_df["value"].value_counts().sort_index().to_dict()
    print("[Events]", event_counts)

    raw = raw.resample(SAMPLE_RATE, verbose=False)
    print(f"[Resample] sfreq = {raw.info['sfreq']} Hz")
    print(f"[Resample] annotations = {len(raw.annotations)}")


    # --------------------------------------------------------
    # 8.3 Optional test crop
    # --------------------------------------------------------
    if TEST:
        raw.crop(tmin=300, tmax=600)

    # --------------------------------------------------------
    # 8.4 Remove invalid EEG channels
    # --------------------------------------------------------
    invalid_present = [ch for ch in INVALID_EEG_CHANNELS if ch in raw.ch_names]
    if invalid_present:
        print("[Drop invalid EEG]:", invalid_present)
        raw.drop_channels(invalid_present)

    # --------------------------------------------------------
    # 8.5 Set channel types
    # --------------------------------------------------------
    channel_type_mapping = {}
    for ch in EOG_CHANNELS:
        if ch in raw.ch_names:
            channel_type_mapping[ch] = "eog"
    if STIM_CHANNEL in raw.ch_names:
        channel_type_mapping[STIM_CHANNEL] = "stim"

    # on_unit_change="ignore" 只是避免：
    # Trigger unit changed from NA to V
    # 这种无害 warning。
    raw.set_channel_types(channel_type_mapping, on_unit_change="ignore")

    # --------------------------------------------------------
    # 8.6 Verify 122 EEG
    # --------------------------------------------------------
    eeg_channels = get_eeg_channels(raw)
    print(f"[Channels] EEG = {len(eeg_channels)}")
    if len(eeg_channels) != 122:
        raise RuntimeError(
            f"Expected 122 EEG channels after removing 10/111, "
            f"got {len(eeg_channels)}."
        )

    # --------------------------------------------------------
    # 8.7 Verify montage matches EEG
    # --------------------------------------------------------
    missing_positions = sorted(set(eeg_channels) - set(MONTAGE.ch_names))
    extra_positions = sorted(set(MONTAGE.ch_names) - set(eeg_channels))

    if missing_positions:
        raise RuntimeError(
            f"EEG channels without montage positions: {missing_positions}"
        )
    if extra_positions:
        raise RuntimeError(
            f"Montage contains channels not present in EEG: {extra_positions}"
        )

    # --------------------------------------------------------
    # 8.8 Set montage
    # --------------------------------------------------------
    raw.set_montage(MONTAGE, on_missing="raise")


    # --------------------------------------------------------
    # 8.9 PREP
    # --------------------------------------------------------
    if PREP:
        print("Running PyPREP ...")
        prep_params = {
            "ref_chs": "eeg",
            "reref_chs": "eeg",
            "line_freqs": np.arange(50, SAMPLE_RATE / 2, 50),
        }
        prep = PrepPipeline(raw, prep_params, MONTAGE, ransac=False)
        prep.fit()
        raw_new = prep.raw.copy()

        # -----------------------------------------------
        # PREP QC
        # -----------------------------------------------
        noisy_original = to_str_list(prep.noisy_channels_original.get("bad_all", [])) # 最初被标记为 bad 的通道列表
        interpolated = to_str_list(prep.interpolated_channels) # 被插值的通道列表
        still_bad = to_str_list(prep.still_noisy_channels) # 插值后仍被标记为 bad 的通道列表

        print()
        print("[PREP] Original bad:", noisy_original)
        print("[PREP] Interpolated:", interpolated)
        print("[PREP] Still bad:", still_bad)
        print("[PREP] raw.info['bads']:", raw_new.info["bads"])
    else:
        print("PREP disabled.")
        raw_new = raw.copy()
        noisy_original = []
        interpolated = []
        still_bad = []

    # --------------------------------------------------------
    # 8.10 High-pass filter
    # --------------------------------------------------------
    # 与 Chisco 1.0 官方 preprocessing 保持一致：
    # PREP 之后额外做 1 Hz high-pass。
    raw_new.filter(l_freq=1.0, h_freq=None, picks="eeg", verbose=False)



    # --------------------------------------------------------
    # 8.11 QC information
    # --------------------------------------------------------
    qc = {
        "edf_file": edf_file.name,
        "events_file": events_file.name,
        "n_channels_raw": 127,
        "n_eeg": len(get_eeg_channels(raw_new)),
        "sfreq": raw_new.info["sfreq"],
        "n_events": len(events_df),
        "n_event_65329": int((events_df["value"] == 65329).sum()),
        "n_event_65379": int((events_df["value"] == 65379).sum()),
        "n_event_65381": int((events_df["value"] == 65381).sum()),
        "original_bad_count": len(noisy_original),
        "original_bad": ";".join(noisy_original),
        "interpolated_count": len(interpolated),
        "interpolated": ";".join(interpolated),
        "still_bad_count": len(still_bad),
        "still_bad": ";".join(still_bad),
    }

    # --------------------------------------------------------
    # 8.12 ICA
    # --------------------------------------------------------

    if ICA:
        ica = mne.preprocessing.ICA(n_components=30, random_state=97, max_iter="auto",method='infomax', fit_params=dict(extended=True)) # 使用extended-infomax算法
        ica.fit(raw_new)
        ic_labels = label_components(raw_new, ica, method="iclabel")
        labels = ic_labels["labels"]
        exclude_idx = [
            idx for idx, label in enumerate(labels) if label not in ["brain", "other"]
        ]
        print(f"Reading Raw Excluding these ICA components: {exclude_idx}")
        raw_new_reconstructed = raw_new.copy()
        ica.apply(raw_new_reconstructed, exclude=exclude_idx)

        # ica.fit 默认 exclude='bads'，PREP 判定的 still-bad 通道（如 POO9h）
        # 没有参与 ICA 拟合。ICA 重建完成后，再用已去伪迹的邻近通道
        # 对这些通道补做一次插值，得到干净的 122 通道数据。
        # interpolate_bads 默认 reset_bads=True，插值后会清空 info['bads']，
        # 避免下游按 bads 再次排除该通道。
        post_ica_bads = list(raw_new_reconstructed.info["bads"])
        if post_ica_bads:
            print(f"[ICA] Interpolate still-bad channels after ICA: {post_ica_bads}")
            raw_new_reconstructed.interpolate_bads()

    # --------------------------------------------------------
    # 8.13 Save continuous FIF
    # --------------------------------------------------------
    output_name = f"{edf_file.stem}_{SUBJECT_ID}_{METHOD_STR}-raw.fif"
    output_path = FIF_DIR / output_name
    raw_new_reconstructed.save(output_path, overwrite=True, verbose=False)
    print(f"[Saved] {output_path}")

    return qc, eeg_channels

def extract_numbers(filename):
    """
    从文件名中提取 para 后的数字和 run 后的数字（去掉前导0）
    
    Args:
        filename: 文件名，如 'sub-01_ses-01_task-para1_run-01_eeg.edf'
    
    Returns:
        tuple: (para编号, run编号) 均为整数
    """
    basename = os.path.basename(filename)
    
    para_match = re.search(r'para(\d+)', basename)
    run_match = re.search(r'run-(\d+)', basename)
    
    para_num = int(para_match.group(1)) if para_match else None
    run_num = int(run_match.group(1)) if run_match else None
    
    return para_num, run_num


# ============================================================
# 9. Traverse dataset
# ============================================================
qc_rows = []
reference_eeg_order = None

for ses_dir in sorted(DATA_FOLDER.glob("ses-*")):
    eeg_dir = ses_dir / "eeg"

    print()
    print("#" * 80)
    print(f"Session: {ses_dir.name}")
    print("#" * 80)

    pairs = pair_edf_and_events(eeg_dir)
    print(f"Found {len(pairs)} EDF/events pairs.")

    for edf_file, events_file in pairs:
        qc, eeg_order = process_edf_file(edf_file, events_file)
        exit(0)

        # ----------------------------------------------------
        # Verify EEG channel order
        # ----------------------------------------------------
        if reference_eeg_order is None:
            reference_eeg_order = eeg_order
            pd.DataFrame(
                {
                    "index": range(len(reference_eeg_order)),
                    "channel": reference_eeg_order,
                }
            ).to_csv(QC_DIR / "eeg_channel_order.csv", index=False)
            print("[Channel order] Reference channel order saved.")
        elif eeg_order != reference_eeg_order:
            raise RuntimeError(
                "\nEEG channel order mismatch!\n"
                f"File: {edf_file}\n"
                "This must be fixed before training CNN/TSConv."
            )

        qc_rows.append(qc)

        # 每处理一个文件就保存一次，
        # 防止长时间运行中断后 QC 全丢失。
        pd.DataFrame(qc_rows).to_csv(QC_DIR / "prep_qc.csv", index=False)


# ============================================================
# 10. Global PREP QC summary
# ============================================================
print()
print("=" * 80)
print("PREPROCESSING FINISHED")
print("=" * 80)

qc_df = pd.DataFrame(qc_rows)
qc_df.to_csv(QC_DIR / "prep_qc.csv", index=False)

still_bad_counter = Counter()
for value in qc_df["still_bad"].fillna(""):
    channels = [ch for ch in value.split(";") if ch]
    still_bad_counter.update(channels)

print()
print(f"Processed recordings: {len(qc_df)}")
print(f"EEG channels per recording: {len(reference_eeg_order)}")

print()
print("Still-bad channel frequency:")
if still_bad_counter:
    for channel, count in still_bad_counter.most_common():
        ratio = count / len(qc_df) * 100
        print(f"  {channel:<10} {count:>3} / {len(qc_df)} ({ratio:.1f}%)")
else:
    print("  No channels remained bad after PREP.")

print()
print(f"QC saved to: {QC_DIR / 'prep_qc.csv'}")
print(f"Channel order saved to: {QC_DIR / 'eeg_channel_order.csv'}")
