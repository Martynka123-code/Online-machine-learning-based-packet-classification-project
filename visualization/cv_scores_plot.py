import os
import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


def plot_cv_scores(
    cv_res,
    reports_dir,
    tag
):
    """Plots cross-validation scores."""

    metrics = {
        "Accuracy": cv_res["test_accuracy"],
        "F1 weighted": cv_res["test_f1_weighted"],
        "F1 macro": cv_res["test_f1_macro"],
    }

    folds = np.arange(1, 6)

    fig, ax = plt.subplots(figsize=(8, 4))

    for name, vals in metrics.items():

        ax.plot(
            folds,
            vals,
            marker="o",
            label=f"{name} (μ={vals.mean():.3f})"
        )

    ax.set_xlabel("Fold")
    ax.set_ylabel("Score")

    ax.set_ylim(0, 1.05)

    ax.set_title(
        "5-Fold Cross-Validation Scores"
    )

    ax.legend()

    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    path = os.path.join(
        reports_dir,
        f"cv_scores_{tag}.png"
    )

    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[+] CV score plot saved → {path}")