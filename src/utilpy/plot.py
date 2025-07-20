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


def plot_roc_curve(fig_path: Path, true_labels: np.ndarray, anomaly_scores: np.ndarray):
    # roc直線
    tpr_list = []
    fpr_list = []
    for ranges in [
        np.arange(1, 10),
        np.arange(10, 100, 10),
        np.arange(100, 1000, 100),
        np.arange(1000, true_labels.shape[0], 500),
    ]:
        for topk in ranges:
            predicted_labels = _get_predicted_labels_by_topk(anomaly_scores, topk)
            fpr, tpr, _ = metrics.roc_curve(true_labels, predicted_labels)
            fpr_list.append(fpr[1])  # 二値予測なのでfpr, tprは[0, 1]の2点のみ
            tpr_list.append(tpr[1])
    plt.figure(figsize=(30, 8))
    plt.plot(fpr_list, tpr_list, label="ROC curve (from varying k)")
    plt.plot([0, 1], [0, 1], "k--", label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve by Varying Top-k")
    plt.legend()
    plt.grid()
    plt.savefig(fig_path)
    plt.close()


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
    return pr_auc, precision, recall, thresholds


def set_major_tick_per_day(
    ax: Axes,
    timeColumn: pd.DatetimeIndex | pd.Series | np.ndarray | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    label_loc: float = -0.03,
    rotation: int = 0,
    format: str = "%m-%d",
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
        ax.axvline(x=tick, color="black", linestyle="--", lw=0.5)  # 縦線
    for label in ax.get_xticklabels():  # 日付ラベルごとに
        label.set_rotation(rotation)  # 回転
        label.set_verticalalignment("top")  # ↓とセットで効く
        label.set_y(label_loc)  # ← y座標を下にずらす（デフォルトより小さい値に）
    return start, end


def set_minor_tick(
    ax: Axes,
    timeColumn: pd.DatetimeIndex | pd.Series | np.ndarray | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
    freq="H",
    format: str = "%H",
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
        ax.axvline(x=tick, color="gray", linestyle="--", lw=0.3)  #
    ax.xaxis.set_minor_formatter(mdates.DateFormatter(format))


def _get_predicted_labels_by_topk(scores, k):
    idx = np.argsort(scores)[-k:]  # 異常スコアが高い順にk個
    predicted = np.zeros_like(scores)
    predicted[idx] = 1
    return predicted
