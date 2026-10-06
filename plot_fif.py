"""
绘制预处理后 FIF 文件中 EEG 波形的简易脚本。

设计目标：
- 只读取并绘制指定的一段数据（由事件区间或时间区间决定），
  不会一次性加载 / 绘制整个文件。
- 通过形参控制：
    * n_channels                       绘制的 EEG 通道数量
    * start_event / end_event / occurrence   由事件描述确定绘制区间
    * tmin / tmax                      直接指定时间区间（秒）

用法示例：
    uv run python plot_fif.py
    uv run python plot_fif.py --channels 8 --start-event "Reading start" --end-event "Inner-speech start"
    uv run python plot_fif.py --channels 10 --tmin 300 --tmax 310
    uv run python plot_fif.py --show
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np

ROOT_FOLDER = Path(__file__).resolve().parent
DEFAULT_FIF = (
    ROOT_FOLDER
    / "preprocess_output"
    / "prep"
    / "sub-01"
    / "fif"
    / "sub-01_ses-01_task-para1_run-01_eeg.fif"
)

# 图片默认保存目录（与 fif 同级），可用 --output 覆盖
PLOT_FOLDER = DEFAULT_FIF.parent.parent / "plot"

# 未指定事件 / 时间区间时，默认绘制多长的一段
DEFAULT_WINDOW = 10.0


def resolve_window(
    raw,
    tmin=None,
    tmax=None,
    start_event=None,
    end_event=None,
    occurrence=0,
    default_duration=DEFAULT_WINDOW,
):
    """确定要绘制的 [tmin, tmax] 时间区间（秒，相对记录起点）。

    优先级：
    1. 显式给定 tmin / tmax 时直接使用（只给 tmin 时右界为 tmin + default_duration）。
    2. 给定 start_event 时，取该描述第 occurrence 次出现作为左界；
       若同时给定 end_event，右界为该左界之后第一次 end_event 的 onset，
       否则右界为左界 + default_duration。
    3. 都没给时，取第一条 annotation 作为起点。
    """
    if tmin is not None or tmax is not None:
        left = 0.0 if tmin is None else float(tmin)
        right = left + default_duration if tmax is None else float(tmax)
        return left, right

    onsets = np.asarray(raw.annotations.onset, dtype=float)
    descriptions = list(raw.annotations.description)

    if start_event is None:
        if len(onsets) == 0:
            return 0.0, min(default_duration, raw.times[-1])
        left = float(onsets[0])
    else:
        matches = [i for i, desc in enumerate(descriptions) if desc == start_event]
        if not matches:
            raise ValueError(
                f"事件描述 {start_event!r} 不存在，可选：{sorted(set(descriptions))}"
            )
        if not 0 <= occurrence < len(matches):
            raise ValueError(
                f"{start_event!r} 共出现 {len(matches)} 次，occurrence={occurrence} 越界。"
            )
        left = float(onsets[matches[occurrence]])

    if end_event is None:
        right = left + default_duration
    else:
        later = [
            onset
            for onset, desc in zip(onsets, descriptions)
            if desc == end_event and onset > left
        ]
        if not later:
            raise ValueError(
                f"{left:.3f}s 之后没有找到 {end_event!r}，"
                f"可选：{sorted(set(descriptions))}"
            )
        right = float(min(later))

    return left, right


def plot_eeg_window(
    fif_file,
    n_channels=10,
    tmin=None,
    tmax=None,
    start_event=None,
    end_event=None,
    occurrence=0,
    channels=None,
    output=None,
    show=False,
):
    """绘制一段 EEG 波形。

    Parameters
    ----------
    fif_file : str | Path
        预处理后的 FIF 文件。
    n_channels : int
        绘制多少个 EEG 通道（从文件里的 EEG 通道中按顺序取前 n 个）。
    tmin, tmax : float | None
        直接指定的时间区间（秒），优先级高于事件区间。
    start_event, end_event : str | None
        用 annotation 描述确定绘制区间，见 resolve_window。
    occurrence : int
        start_event 取第几次出现（从 0 开始）。
    channels : list[str] | None
        显式指定要绘制的通道名；给定时忽略 n_channels。
    output : str | Path | None
        图片保存路径（仅在 show=False 时生效）；
        默认保存到 plot 目录下 <文件名>_plot.png。
    show : bool
        为 True 时只弹出交互窗口、不保存图片；
        为 False 时才保存图片到 output，不弹窗。
    """
    fif_file = Path(fif_file)
    raw = mne.io.read_raw_fif(fif_file, preload=False, verbose=False)

    eeg_names = [
        name
        for name, ch_type in zip(raw.ch_names, raw.get_channel_types())
        if ch_type == "eeg"
    ]
    selected = list(channels) if channels is not None else eeg_names[:n_channels]
    if not selected:
        raise ValueError("没有可绘制的 EEG 通道。")

    left, right = resolve_window(
        raw, tmin, tmax, start_event, end_event, occurrence
    )
    left = max(left, 0.0)
    right = min(right, raw.times[-1])
    if right <= left:
        raise ValueError(f"无效的绘制区间：[{left}, {right}]")

    # 先记录窗口内的事件（绝对 onset），crop 后 x 轴以 0 为起点
    events_in_window = [
        (float(onset), desc)
        for onset, desc in zip(raw.annotations.onset, raw.annotations.description)
        if left <= onset <= right
    ]

    # 只挑选需要的通道并裁剪时间，再加载这一小段数据
    raw.pick(selected)
    raw.crop(tmin=left, tmax=right)
    raw.load_data()

    data = raw.get_data() * 1e6  # V -> µV
    # crop 后 raw.times 从 0 重新计时，这里还原成相对记录起点的绝对时间，
    # 与 xlim / 事件竖线保持同一坐标。
    times = raw.times + left
    n_plot = len(raw.ch_names)

    peak = float(np.abs(data).max()) if data.size else 1.0
    separation = max(peak * 2.0, 1.0)

    fig, ax = plt.subplots(figsize=(14, max(3.0, 0.45 * n_plot)))
    for i, ch_name in enumerate(raw.ch_names):
        ax.plot(times, data[i] + i * separation, lw=0.6, color="black")

    ax.set_yticks(np.arange(n_plot) * separation)
    ax.set_yticklabels(raw.ch_names, fontsize=8)
    ax.set_xlim(left, right)
    ax.set_xlabel("Time (s)")
    ax.set_title(
        f"{fif_file.stem}\n"
        f"{left:.2f} - {right:.2f} s | {n_plot} EEG channels"
    )

    for onset, desc in events_in_window:
        ax.axvline(onset, color="red", ls="--", lw=0.8, alpha=0.7)
        ax.text(
            onset,
            n_plot * separation,
            desc,
            rotation=90,
            va="bottom",
            ha="right",
            fontsize=7,
            color="red",
        )

    fig.tight_layout()

    # show=True：只弹窗查看，不落盘
    if show:
        plt.show()
        return None

    # show=False：保存图片，不弹窗
    if output is None:
        output = PLOT_FOLDER / f"{fif_file.stem}_plot.png"
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    print(f"[Saved] {output}")

    return output


def main():
    parser = argparse.ArgumentParser(
        description="绘制预处理 FIF 中一段 EEG 波形（不会一次性绘制全部数据）。"
    )
    parser.add_argument("--file", default=str(DEFAULT_FIF), help="FIF 文件路径")
    parser.add_argument("--channels", type=int, default=10, help="绘制的 EEG 通道数量")
    parser.add_argument("--tmin", type=float, default=None, help="起始时间（秒）")
    parser.add_argument("--tmax", type=float, default=None, help="结束时间（秒）")
    parser.add_argument("--start-event", default=None, help="起始事件描述")
    parser.add_argument("--end-event", default=None, help="结束事件描述")
    parser.add_argument("--occurrence", type=int, default=0, help="start-event 第几次出现（从 0 开始）")
    parser.add_argument("--output", default=None, help="图片保存路径（show=False 时生效）")
    parser.add_argument("--show", action="store_true", help="只弹窗查看，不保存图片")
    args = parser.parse_args()

    plot_eeg_window(
        fif_file=args.file,
        n_channels=args.channels,
        tmin=args.tmin,
        tmax=args.tmax,
        start_event=args.start_event,
        end_event=args.end_event,
        occurrence=args.occurrence,
        output=args.output,
        show=args.show,
    )


if __name__ == "__main__":
    main()