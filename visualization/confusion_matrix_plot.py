import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, labels, reports_dir, tag):
    """Generuje nowoczesną macierz pomyłek, odporną na błędy typów danych."""
    
    # Konwertujemy y_true i y_pred do czystych struktur numpy intów
    y_true_arr = np.array(y_true, dtype=int)
    y_pred_arr = np.array(y_pred, dtype=int)
    
    # Pobieramy unikalne numery klas, które FIZYCZNIE wystąpiły w y_true
    unique_in_true = np.unique(y_true_arr)
    
    # Jeśli y_true jest puste (skrajny przypadek), przerywamy bezpiecznie
    if len(unique_in_true) == 0:
        print("[!] Brak danych w y_true — pomijam generowanie macierzy pomyłek.")
        return

    # Obliczamy macierz pomyłek TYLKO dla klas obecnych w y_true, żeby uniknąć crasha sklearn
    cm_raw = confusion_matrix(y_true_arr, y_pred_arr, labels=unique_in_true)

    # Odtwarzamy pełnowymiarową macierz (n_classes x n_classes), aby wykres zawierał wszystkie aplikacje
    n_classes = len(labels)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    
    for i, true_idx in enumerate(unique_in_true):
        for j in range(len(unique_in_true)):
            # Mapujemy małą macierz z powrotem na pełne indeksy klas
            pred_idx = unique_in_true[j]
            if pred_idx < n_classes and true_idx < n_classes:
                cm[true_idx, pred_idx] = cm_raw[i, j]

    # Bezpieczna normalizacja (Recall)
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        cm_norm = np.nan_to_num(cm_norm)

    # Dynamiczne dopasowanie rozmiaru wykresu do liczby klas
    fig_width = max(13, n_classes * 1.0)
    fig_height = max(5.5, n_classes * 0.5)
    
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height))
    sns.set_theme(style="white")

    annot_size = 11 if n_classes <= 10 else (8 if n_classes <= 20 else 6)

    # --- Wykres 1: Wartości surowe ---
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=labels, yticklabels=labels, cbar=False,
                annot_kws={"size": annot_size})
    
    axes[0].set_title("Macierz Pomyłek — Liczba Wystąpień", fontsize=13, fontweight='bold', pad=12)
    axes[0].set_xlabel("Klasa Przewidywana", fontsize=11)
    axes[0].set_ylabel("Klasa Rzeczywista", fontsize=11)
    axes[0].tick_params(axis="x", rotation=30, labelsize=9)
    axes[0].tick_params(axis="y", rotation=0, labelsize=9)

    # --- Wykres 2: Wartości znormalizowane ---
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", ax=axes[1],
                xticklabels=labels, yticklabels=labels, cbar=True,
                annot_kws={"size": annot_size})
    
    axes[1].set_title("Macierz Pomyłek — Znormalizowana (Recall)", fontsize=13, fontweight='bold', pad=12)
    axes[1].set_xlabel("Klasa Przewidywana", fontsize=11)
    axes[1].set_ylabel("Klasa Rzeczywista", fontsize=11)
    axes[1].tick_params(axis="x", rotation=30, labelsize=9)
    axes[1].tick_params(axis="y", rotation=0, labelsize=9)

    plt.tight_layout()
    
    path = os.path.join(reports_dir, f"confusion_matrix_{tag}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[+] Macierz pomyłek została pomyślnie zapisana → {path}")
