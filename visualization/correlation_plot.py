import os
import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns


def plot_feature_correlation(
    X_before,
    X_after,
    reports_dir,
    tag
):
    """Plots correlation heatmaps before vs after feature removal."""

    corr_before = X_before.corr(method="pearson")
    corr_after = X_after.corr(method="pearson")

    n_before = len(corr_before)
    n_after = len(corr_after)

    cell_size = 0.85

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(
            n_before * cell_size +
            n_after * cell_size + 2,
            max(n_before, n_after) *
            cell_size + 2
        )
    )

    for ax, corr, title in [
        (axes[0], corr_before, f"BEFORE ({n_before} features)"),
        (axes[1], corr_after, f"AFTER ({n_after} features)")
    ]:

        mask = np.triu(
            np.ones_like(corr, dtype=bool)
        )

        sns.heatmap(
            corr,
            mask=mask,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=0.5,
            annot_kws={"size": 7},
            ax=ax,
            vmin=-1,
            vmax=1,
            cbar=False
        )

        ax.set_title(title)

        ax.tick_params(
            axis="x",
            rotation=45,
            labelsize=7
        )

        ax.tick_params(
            axis="y",
            rotation=0,
            labelsize=7
        )

    plt.tight_layout()

    path = os.path.join(
        reports_dir,
        f"feature_correlation_{tag}.png"
    )

    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[+] Correlation matrix saved → {path}")