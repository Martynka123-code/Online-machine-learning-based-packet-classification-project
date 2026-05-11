import os
import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay
)


def plot_confusion_matrix(
    y_true,
    y_pred,
    labels,
    reports_dir,
    tag
):
    """Plots raw + normalized confusion matrix."""

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw counts
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels
    )

    disp.plot(
        ax=axes[0],
        colorbar=False,
        cmap="Blues"
    )

    axes[0].set_title("Confusion Matrix — raw counts")
    axes[0].tick_params(axis="x", rotation=45)

    # Normalized
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    disp2 = ConfusionMatrixDisplay(
        confusion_matrix=cm_norm,
        display_labels=labels
    )

    disp2.plot(
        ax=axes[1],
        colorbar=False,
        cmap="Blues",
        values_format=".2f"
    )

    axes[1].set_title("Confusion Matrix — normalized recall")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()

    path = os.path.join(
        reports_dir,
        f"confusion_matrix_{tag}.png"
    )

    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[+] Confusion matrix saved → {path}")