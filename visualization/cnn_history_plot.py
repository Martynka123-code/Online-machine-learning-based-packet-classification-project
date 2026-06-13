import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_training_history(csv_path="lightning_logs/version_0/metrics.csv"):
    """Rysuje krzywe uczenia (Loss i F1/Accuracy) z logów PyTorch Lightning."""
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"[!] Nie znaleziono {csv_path}.")
        return

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Wyciągnięcie danych (Lightning loguje train/val w różnych wierszach, więc ffill pomaga to połączyć)
    df_plot = df.ffill().bfill()

    # Plot 1: Loss
    if 'train_loss' in df_plot.columns and 'val_loss' in df_plot.columns:
        sns.lineplot(data=df_plot, x='epoch', y='train_loss', label='Train Loss', ax=axes[0], color='blue', linewidth=2)
        sns.lineplot(data=df_plot, x='epoch', y='val_loss', label='Val Loss', ax=axes[0], color='red', linewidth=2)
        axes[0].set_title("Krzywa Funkcji Straty (Loss)", fontsize=14)
        axes[0].set_ylabel("Cross Entropy Loss")

    # Plot 2: Accuracy / F1
    if 'val_acc' in df_plot.columns and 'val_f1_macro' in df_plot.columns:
        sns.lineplot(data=df_plot, x='epoch', y='val_acc', label='Val Accuracy', ax=axes[1], color='green', linewidth=2)
        sns.lineplot(data=df_plot, x='epoch', y='val_f1_macro', label='Val F1-Macro', ax=axes[1], color='purple', linewidth=2)
        axes[1].set_title("Skuteczność Modelu (Accuracy / F1)", fontsize=14)
        axes[1].set_ylabel("Wartość metryki")

    plt.tight_layout()
    plt.savefig("cnn_training_history.png", dpi=300)
    print("[+] Krzywe uczenia zapisane jako cnn_training_history.png")