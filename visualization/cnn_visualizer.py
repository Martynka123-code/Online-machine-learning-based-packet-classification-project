import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def plot_cnn_training_curves(
    train_losses,
    val_losses,
    val_accuracies,
    reports_dir,
    tag="cnn",
    val_f1_scores=None
):
    """
    Plots CNN training loss curves, validation accuracy, and optional F1 macro.

    Lightning runs a sanity-check validation before epoch 0, so val lists
    may have one extra entry — all series are trimmed to the shortest common
    length before plotting.

    Parameters
    ----------
    train_losses : list[float]
    val_losses : list[float]
    val_accuracies : list[float]
    reports_dir : str
    tag : str
    val_f1_scores : list[float] | None
        If provided, adds a third F1 panel.
    """
    # Trim to shortest common length (fixes Lightning sanity-check offset)
    n = min(len(train_losses), len(val_losses), len(val_accuracies))
    if val_f1_scores:
        n = min(n, len(val_f1_scores))
        val_f1_scores = val_f1_scores[-n:]

    train_losses   = train_losses[-n:]
    val_losses     = val_losses[-n:]
    val_accuracies = val_accuracies[-n:]
    epochs = list(range(1, n + 1))

    n_cols = 3 if val_f1_scores else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 4))
    ax1, ax2 = axes[0], axes[1]

    # --- Loss ---
    ax1.plot(epochs, train_losses, marker="o", linewidth=2,
             color="#2196F3", label="Train loss")
    ax1.plot(epochs, val_losses, marker="s", linewidth=2,
             color="#F44336", linestyle="--", label="Val loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-entropy loss")
    ax1.set_title("Training vs Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Accuracy ---
    ax2.plot(epochs, [a * 100 for a in val_accuracies],
             marker="o", linewidth=2, color="#4CAF50")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Validation Accuracy per Epoch")
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3)

    for i, acc in enumerate(val_accuracies):
        ax2.annotate(
            f"{acc * 100:.1f}%",
            (epochs[i], acc * 100),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8
        )

    # --- F1 macro (optional third panel) ---
    if val_f1_scores:
        ax3 = axes[2]
        ax3.plot(epochs, [f * 100 for f in val_f1_scores],
                 marker="^", linewidth=2, color="#FF9800")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("F1 macro (%)")
        ax3.set_title("Validation F1 Macro per Epoch")
        ax3.set_ylim(0, 105)
        ax3.grid(True, alpha=0.3)

        for i, f1 in enumerate(val_f1_scores):
            ax3.annotate(
                f"{f1 * 100:.1f}%",
                (epochs[i], f1 * 100),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8
            )

    plt.tight_layout()

    path = os.path.join(reports_dir, f"cnn_training_curves_{tag}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[+] CNN training curves saved → {path}")


def plot_cnn_class_distribution(
    labels,
    class_map,
    reports_dir,
    tag="cnn"
):
    """
    Plots class distribution of the CNN dataset.
    """
    idx_to_app = {v: k for k, v in class_map.items()}
    unique, counts = np.unique(labels, return_counts=True)
    class_names = [idx_to_app.get(int(u), str(u)) for u in unique]

    fig, ax = plt.subplots(figsize=(max(6, len(unique) * 1.2), 4))

    colors = plt.cm.tab10(np.linspace(0, 1, len(unique)))
    bars = ax.bar(class_names, counts, color=colors, edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Application class")
    ax.set_ylabel("Sample count")
    ax.set_title("CNN Dataset — class distribution")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y", alpha=0.3)

    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.01,
            str(count),
            ha="center",
            va="bottom",
            fontsize=9
        )

    plt.tight_layout()

    path = os.path.join(reports_dir, f"cnn_class_distribution_{tag}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[+] CNN class distribution saved → {path}")


def plot_cnn_confusion_matrix(
    y_true,
    y_pred,
    class_map,
    reports_dir,
    tag="cnn"
):
    """
    Plots raw and normalized confusion matrix for CNN predictions.
    """
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    idx_to_app = {v: k for k, v in class_map.items()}
    labels_idx = sorted(idx_to_app.keys())
    label_names = [idx_to_app[i] for i in labels_idx]

    cm = confusion_matrix(y_true, y_pred, labels=labels_idx)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    disp.plot(ax=axes[0], colorbar=False, cmap="Blues")
    axes[0].set_title("CNN Confusion Matrix — raw counts")
    axes[0].tick_params(axis="x", rotation=45)

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=label_names)
    disp2.plot(ax=axes[1], colorbar=False, cmap="Blues", values_format=".2f")
    axes[1].set_title("CNN Confusion Matrix — normalized recall")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()

    path = os.path.join(reports_dir, f"cnn_confusion_matrix_{tag}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[+] CNN confusion matrix saved → {path}")


def plot_cnn_f1_per_class(
    y_true,
    y_pred,
    class_map,
    reports_dir,
    tag="cnn"
):
    """
    Plots per-class Precision / F1 / Recall as a grouped bar chart.
    """
    from sklearn.metrics import classification_report

    idx_to_app = {v: k for k, v in class_map.items()}
    labels_idx = sorted(idx_to_app.keys())
    label_names = [idx_to_app[i] for i in labels_idx]

    report = classification_report(
        y_true, y_pred,
        labels=labels_idx,
        target_names=label_names,
        output_dict=True,
        zero_division=0
    )

    classes   = [n for n in label_names if n in report]
    f1_scores = [report[n]["f1-score"]  for n in classes]
    precision = [report[n]["precision"] for n in classes]
    recall    = [report[n]["recall"]    for n in classes]
    macro_f1  = report.get("macro avg", {}).get("f1-score", 0.0)

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(8, len(classes) * 1.5), 5))

    ax.bar(x - width, precision, width, label="Precision", color="#2196F3", alpha=0.85)
    ax.bar(x,         f1_scores, width, label="F1 score",  color="#FF9800", alpha=0.85)
    ax.bar(x + width, recall,    width, label="Recall",    color="#4CAF50", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.12)
    ax.set_title(f"Per-class Precision / F1 / Recall  (macro F1 = {macro_f1:.3f})")
    ax.axhline(macro_f1, color="#F44336", linestyle="--", linewidth=1.2,
               label=f"Macro F1 = {macro_f1:.3f}")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    for i, (p, f, r) in enumerate(zip(precision, f1_scores, recall)):
        for offset, val in [(-width, p), (0, f), (width, r)]:
            ax.text(i + offset, val + 0.01, f"{val:.2f}",
                    ha="center", va="bottom", fontsize=7)

    plt.tight_layout()

    path = os.path.join(reports_dir, f"cnn_f1_per_class_{tag}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[+] CNN per-class F1 chart saved → {path}")