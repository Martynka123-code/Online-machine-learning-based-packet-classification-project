import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


def plot_granularity_results(results, output_dir="reports"):
    """Plots accuracy vs granularity."""

    grans = sorted(results.keys())
    accs = [results[g] * 100 for g in grans]

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(
        grans,
        accs,
        marker="o",
        linewidth=2,
        color="#2196F3"
    )

    ax.fill_between(
        grans,
        accs,
        alpha=0.15,
        color="#2196F3"
    )

    ax.set_xlabel("Granularity (packets per window)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy vs Granularity")

    ax.grid(True, alpha=0.3)

    for g, a in zip(grans, accs):
        ax.annotate(
            f"{a:.1f}%",
            (g, a),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8
        )

    plt.tight_layout()

    path = os.path.join(
        output_dir,
        "granularity_comparison.png"
    )

    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[+] Granularity comparison chart saved → {path}")