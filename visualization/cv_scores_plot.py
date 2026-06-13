import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_cv_scores(cv_res, reports_dir, tag):
    """
    Rysuje wyniki cross-walidacji.
    POPRAWKA: folds obliczane dynamicznie z długości danych, nie hardkodowane jako 1..6.
    """
    metrics = {
        "Accuracy": cv_res["test_accuracy"],
        "F1 weighted": cv_res["test_f1_weighted"],
        "F1 macro": cv_res["test_f1_macro"],
    }

    # Dynamicznie — ile foldów faktycznie mamy w danych
    n_folds = len(cv_res["test_accuracy"])
    folds = np.arange(1, n_folds + 1)

    fig, ax = plt.subplots(figsize=(8, 4))

    for name, vals in metrics.items():
        vals = np.array(vals)
        ax.plot(
            folds[:len(vals)],
            vals,
            marker="o",
            label=f"{name} (μ={vals.mean():.3f})"
        )

    ax.set_xlabel("Fold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(folds)
    ax.set_title(f"{n_folds}-Fold Cross-Validation Scores")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    path = os.path.join(reports_dir, f"cv_scores_{tag}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[+] CV score plot saved → {path}")