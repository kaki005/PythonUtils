from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from sklearn import metrics


def plot_pr_curve(fig_path: Path, true_labels: np.ndarray, anomaly_scores: np.ndarray):
    # Pr-auc直線
    precision, recall, thresholds = metrics.precision_recall_curve(true_labels, anomaly_scores)
    pr_auc = metrics.auc(recall, precision)
    plt.plot(recall, precision, label="PR curve (area = %.2f)" % pr_auc)
    plt.legend()
    plt.title("PR curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True)
    plt.savefig(fig_path)
    plt.close("all")
    return pr_auc, precision, recall, thresholds


def set_major_tick_per_year(
    ax: Axes,
    timeColumn: pd.DatetimeIndex | pd.Series | np.ndarray | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    label_loc: float = -0.03,
    rotation: int = 0,
    format: str = "%y",
    tick_collor: str = "black",
    tick_linestyle: str = "--",
):
    if timeColumn is not None and isinstance(timeColumn, np.ndarray):
        timeColumn = pd.Series(timeColumn)
    if start is None:
        assert timeColumn is not None
        start = (
            timeColumn.min().replace(year=1, day=1, hour=0, minute=0, second=0, microsecond=0).to_pydatetime()
        )  # 開始を日付の始まりに設定
    if end is None:
        assert timeColumn is not None
        end = pd.to_datetime(timeColumn.max()).to_pydatetime()
    dayLocator = mdates.YearLocator()
    ax.xaxis.set_major_locator(dayLocator)
    ax.xaxis.set_major_formatter(mdates.DateFormatter(format))
    for tick in dayLocator.tick_values(start, end):  # 日付の変わり目に
        ax.axvline(x=tick, color=tick_collor, linestyle=tick_linestyle, lw=0.5)
    _set_major_tick_pos(ax, label_loc, rotation)
    return start, end


def set_major_tick_per_month(
    ax: Axes,
    timeColumn: pd.DatetimeIndex | pd.Series | np.ndarray | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    label_loc: float = -0.03,
    rotation: int = 0,
    format: str = "%m",  # Changed to year-month format
    tick_collor: str = "black",
    tick_linestyle: str = "--",
):
    """
    Sets the major ticks on the x-axis to the beginning of each month.
    """
    if timeColumn is not None and isinstance(timeColumn, np.ndarray):
        timeColumn = pd.Series(timeColumn)
    if start is None:
        assert timeColumn is not None
        start = timeColumn.min().replace(day=1, hour=0, minute=0, second=0, microsecond=0).to_pydatetime()
    if end is None:
        assert timeColumn is not None
        end = pd.to_datetime(timeColumn.max()).to_pydatetime()

    # Use MonthLocator instead of DayLocator
    monthLocator = mdates.MonthLocator()
    ax.xaxis.set_major_locator(monthLocator)
    ax.xaxis.set_major_formatter(mdates.DateFormatter(format))
    # Draw vertical lines at the beginning of each month
    for tick in monthLocator.tick_values(start, end):
        ax.axvline(x=tick, color=tick_collor, linestyle=tick_linestyle, lw=0.5)
    _set_major_tick_pos(ax, label_loc, rotation)
    return start, end


def set_major_tick_per_day(
    ax: Axes,
    timeColumn: pd.DatetimeIndex | pd.Series | np.ndarray | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    label_loc: float = -0.03,
    rotation: int = 0,
    format: str = "%m-%d",
    tick_collor: str = "black",
    tick_linestyle: str = "--",
):
    if timeColumn is not None and isinstance(timeColumn, np.ndarray):
        timeColumn = pd.Series(timeColumn)
    if start is None:
        assert timeColumn is not None
        start = (
            timeColumn.min().replace(hour=0, minute=0, second=0, microsecond=0).to_pydatetime()
        )  # 開始を日付の始まりに設定
    if end is None:
        assert timeColumn is not None
        end = pd.to_datetime(timeColumn.max()).to_pydatetime()
    dayLocator = mdates.DayLocator()
    ax.xaxis.set_major_locator(dayLocator)
    ax.xaxis.set_major_formatter(mdates.DateFormatter(format))
    for tick in dayLocator.tick_values(start, end):  # 日付の変わり目に
        ax.axvline(x=tick, color=tick_collor, linestyle=tick_linestyle, lw=0.5)
    _set_major_tick_pos(ax, label_loc, rotation)
    return start, end


def set_major_tick_fixed(
    ax: Axes,
    timeColumn: pd.DatetimeIndex | pd.Series | np.ndarray | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    freq="H",
    format: str = "%H",
    tick_collor: str = "gray",
    tick_linestyle: str = "--",
):
    if timeColumn is not None and isinstance(timeColumn, np.ndarray):
        timeColumn = pd.Series(timeColumn)
    if start is None:
        assert timeColumn is not None
        start = (
            timeColumn.min().replace(hour=0, minute=0, second=0, microsecond=0).to_pydatetime()
        )  # 開始を日付の始まりに設定
    if end is None:
        assert timeColumn is not None
        end = pd.to_datetime(timeColumn.max()).to_pydatetime()
    major_ticks = pd.date_range(start=start, end=end, freq=freq)
    major_locator = ticker.FixedLocator(major_ticks.map(mdates.date2num))
    ax.xaxis.set_major_locator(major_locator)
    for tick in major_locator.tick_values(start, end):
        ax.axvline(x=tick, color=tick_collor, linestyle=tick_linestyle, lw=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter(format))


def _set_major_tick_pos(
    ax,
    label_loc: float = -0.03,
    rotation: int = 0,
):
    for label in ax.get_xticklabels():  # 日付ラベルごとに
        label.set_rotation(rotation)  # 回転
        label.set_verticalalignment("top")  # ↓とセットで効く
        label.set_y(label_loc)  # ← y座標を下にずらす（デフォルトより小さい値に）


def set_minor_tick(
    ax: Axes,
    timeColumn: pd.DatetimeIndex | pd.Series | np.ndarray | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    freq="H",
    format: str = "%H",
    tick_collor: str = "gray",
    tick_linestyle: str = "--",
):
    if timeColumn is not None and isinstance(timeColumn, np.ndarray):
        timeColumn = pd.Series(timeColumn)
    if start is None:
        assert timeColumn is not None
        start = (
            timeColumn.min().replace(hour=0, minute=0, second=0, microsecond=0).to_pydatetime()
        )  # 開始を日付の始まりに設定
    if end is None:
        assert timeColumn is not None
        end = pd.to_datetime(timeColumn.max()).to_pydatetime()
    minor_ticks = pd.date_range(start=start, end=end, freq=freq)
    minor_locator = ticker.FixedLocator(minor_ticks.map(mdates.date2num))
    ax.xaxis.set_minor_locator(minor_locator)
    for tick in minor_locator.tick_values(start, end):
        ax.axvline(x=tick, color=tick_collor, linestyle=tick_linestyle, lw=0.5)
    ax.xaxis.set_minor_formatter(mdates.DateFormatter(format))
