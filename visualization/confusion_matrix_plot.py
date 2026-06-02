import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, labels, reports_dir, tag):
    """Plots raw + normalized confusion matrix using display text labels."""
    
    # labels zawiera stringi (np. ['Klasa 0', 'Klasa 1'...])
    # Ponieważ y_true i y_pred zawierają liczby (0, 1, 2...), informujemy scikit-learn,
    # aby wygenerował macierz dla indeksów od 0 do N-1.
    n_classes = len(labels)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))

    # Bezpieczna normalizacja (unikamy dzielenia przez zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)

    # Dostosowanie rozmiaru okna do liczby klas
    fig_width = max(14, n_classes * 0.9)
    fig_height = max(6, n_classes * 0.45)
    
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height))
    sns.set_theme(style="white")

    annot_size = 12 if n_classes <= 10 else (8 if n_classes <= 20 else 6)

    # 1. Macierz wartości bezwzględnych (Raw counts)
    # Przekazujemy tekstowe 'labels' jako xticklabels i yticklabels do Seaborna
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=labels, yticklabels=labels, cbar=False,
                annot_kws={"size": annot_size})
    
    axes[0].set_title("Macierz Pomyłek — Liczba Wystąpień", fontsize=14, fontweight='bold', pad=12)
    axes[0].set_xlabel("Klasa Przewidywana", fontsize=11)
    axes[0].set_ylabel("Klasa Rzeczywista", fontsize=11)
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].tick_params(axis="y", rotation=0)

    # 2. Macierz znormalizowana (Normalized recall)
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", ax=axes[1],
                xticklabels=labels, yticklabels=labels, cbar=True,
                annot_kws={"size": annot_size})
    
    axes[1].set_title("Macierz Pomyłek — Znormalizowana (Recall)", fontsize=14, fontweight='bold', pad=12)
    axes[1].set_xlabel("Klasa Przewidywana", fontsize=11)
    axes[1].set_ylabel("Klasa Rzeczywista", fontsize=11)
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].tick_params(axis="y", rotation=0)

    plt.tight_layout()
    
    path = os.path.join(reports_dir, f"confusion_matrix_{tag}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[+] Macierz pomyłek została pomyślnie zapisana → {path}")