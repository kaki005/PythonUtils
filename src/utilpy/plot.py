from pathlib import Path

import matplotlib.pyplot as plt
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


def _get_predicted_labels_by_topk(scores, k):
    idx = np.argsort(scores)[-k:]  # 異常スコアが高い順にk個
    predicted = np.zeros_like(scores)
    predicted[idx] = 1
    return predicted
