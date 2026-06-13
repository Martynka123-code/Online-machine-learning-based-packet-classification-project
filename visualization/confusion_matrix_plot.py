import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true, y_pred, labels, reports_dir, tag):
    """
    Generuje macierz pomyłek.
    POPRAWKA: y_true/y_pred to stringi (nazwy klas), nie inty.
    Nie konwertujemy do int — sklearn obsługuje string labels natywnie.
    """
    if len(y_true) == 0:
        print("[!] Brak danych w y_true — pomijam generowanie macierzy pomyłek.")
        return

    # Upewniamy się że labels to posortowana lista unikalnych klas (stringi)
    all_labels = sorted(set(list(y_true) + list(y_pred)))
    # Jeśli przekazano labels z zewnątrz, używamy ich (mogą zawierać klasy bez próbek w teście)
    if labels is not None and len(labels) > 0:
        # Zachowaj tylko klasy które faktycznie wystąpiły, w podanej kolejności
        display_labels = [l for l in labels if l in all_labels]
        # Dodaj klasy z predykcji których nie było w labels (np. "OTHER")
        for l in all_labels:
            if l not in display_labels:
                display_labels.append(l)
    else:
        display_labels = all_labels

    cm = confusion_matrix(y_true, y_pred, labels=display_labels)

    # Bezpieczna normalizacja (Recall)
    with np.errstate(divide='ignore', invalid='ignore'):
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.where(row_sums > 0, cm.astype(float) / row_sums, 0.0)

    n_classes = len(display_labels)
    fig_width = max(13, n_classes * 1.0)
    fig_height = max(5.5, n_classes * 0.5)

    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height))
    sns.set_theme(style="white")

    annot_size = 11 if n_classes <= 10 else (8 if n_classes <= 20 else 6)

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=display_labels, yticklabels=display_labels,
                cbar=False, annot_kws={"size": annot_size})
    axes[0].set_title("Macierz Pomyłek — Liczba Wystąpień", fontsize=13, fontweight='bold', pad=12)
    axes[0].set_xlabel("Klasa Przewidywana", fontsize=11)
    axes[0].set_ylabel("Klasa Rzeczywista", fontsize=11)
    axes[0].tick_params(axis="x", rotation=30, labelsize=9)
    axes[0].tick_params(axis="y", rotation=0, labelsize=9)

    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", ax=axes[1],
                xticklabels=display_labels, yticklabels=display_labels,
                cbar=True, annot_kws={"size": annot_size})
    axes[1].set_title("Macierz Pomyłek — Znormalizowana (Recall)", fontsize=13, fontweight='bold', pad=12)
    axes[1].set_xlabel("Klasa Przewidywana", fontsize=11)
    axes[1].set_ylabel("Klasa Rzeczywista", fontsize=11)
    axes[1].tick_params(axis="x", rotation=30, labelsize=9)
    axes[1].tick_params(axis="y", rotation=0, labelsize=9)

    plt.tight_layout()

    path = os.path.join(reports_dir, f"confusion_matrix_{tag}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[+] Macierz pomyłek zapisana → {path}")