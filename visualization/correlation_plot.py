import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_correlation(X_before, X_after, reports_dir, tag):
    """Plots correlation heatmaps before vs after feature removal."""

    corr_before = X_before.corr(method="pearson")
    corr_after = X_after.corr(method="pearson")

    n_before = len(corr_before)
    n_after = len(corr_after)

    cell_size = 0.85

    # Obliczanie szerokości z uwzględnieniem miejsca na pojedynczy cbar (+1)
    fig_width = n_before * cell_size + n_after * cell_size + 3
    fig_height = max(n_before, n_after) * cell_size + 2

    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height))
    
    # Zabezpieczenie: spłaszczamy axes do 1D, na wypadek nietypowego zachowania matplotlib
    axes = axes.flatten()

    # Konfiguracja rysowania dla obu macierzy
    plots_config = [
        (axes[0], corr_before, f"BEFORE ({n_before} features)", False),
        (axes[1], corr_after, f"AFTER ({n_after} features)", True) # Włączamy cbar tylko dla drugiego
    ]

    for ax, corr, title, show_cbar in plots_config:
        # Maska dla dolnego trójkąta (ukrywamy górny trójkąt korelacji, bo jest lustrzany)
        mask = np.triu(np.ones_like(corr, dtype=bool))

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
            cbar=show_cbar,
            cbar_kws={"shrink": 0.7} if show_cbar else None # Estetyczne zmniejszenie paska
        )

        ax.set_title(title, fontsize=11, fontweight='bold', pad=10)

        # Poprawione: ha='right' sprawia, że skośne napisy idealnie trafiają w środek kolumny
        ax.set_xticklabels(
            ax.get_xticklabels(), 
            rotation=45, 
            horizontalalignment='right', 
            fontsize=7
        )
        
        ax.tick_params(
            axis="y",
            rotation=0,
            labelsize=7
        )

    plt.tight_layout()

    # Tworzenie katalogu na raporty, jeśli nie istnieje
    os.makedirs(reports_dir, exist_ok=True)

    path = os.path.join(reports_dir, f"feature_correlation_{tag}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"[+] Correlation matrix saved → {path}")