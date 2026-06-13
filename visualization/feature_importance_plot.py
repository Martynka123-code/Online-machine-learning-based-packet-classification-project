import os
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


def plot_feature_importance(
    model,
    feature_names,
    reports_dir,
    tag
):
    """Plots Random Forest feature importances."""

    importances = pd.Series(
        model.feature_importances_,
        index=feature_names
    ).sort_values(ascending=False)

    fig, ax = plt.subplots(
        figsize=(10, max(4, len(importances) * 0.45))
    )

    colors = [
        "#2196F3" if i < 5 else "#90CAF9"
        for i in range(len(importances))
    ]

    importances.plot(
        kind="barh",
        ax=ax,
        color=colors[::-1]
    )

    ax.invert_yaxis()

    ax.set_xlabel(
        "Mean decrease in impurity (Gini importance)"
    )

    ax.set_title("Feature Importance (Random Forest)")

    ax.axvline(
        importances.mean(),
        color="red",
        linestyle="--",
        label="mean"
    )

    ax.legend()

    plt.tight_layout()

    path = os.path.join(
        reports_dir,
        f"feature_importance_{tag}.png"
    )

    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[+] Feature importance saved → {path}")

    print("\nTop-10 features:")

    for feat, val in importances.head(10).items():
        bar = "█" * int(val * 200)

        print(
            f"  {feat:30s}: "
            f"{val:.5f} {bar}"
        )