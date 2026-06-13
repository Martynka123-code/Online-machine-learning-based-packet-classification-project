import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def compare_datasets(csv_path_train, csv_path_test, app_name, top_features_to_plot=6):
    """
    Porównuje rozkłady cech dla tej samej aplikacji pomiędzy zbiorem treningowym, 
    a zbiorem nagranym później (testowym).
    """
    print(f"[*] Ładowanie danych dla aplikacji: {app_name}")
    
    # Ładujemy CSV
    df_train = pd.read_csv(csv_path_train)
    df_test = pd.read_csv(csv_path_test)
    
    # Filtrujemy tylko interesującą nas aplikację (jeśli pliki zawierają wiele etykiet)
    if "label" in df_train.columns:
        df_train = df_train[df_train["label"] == app_name]
        df_test = df_test[df_test["label"] == app_name]

    df_train["Zbiór"] = "Treningowy (Stary)"
    df_test["Zbiór"] = "Testowy (Nowy)"

    # Łączymy w jeden DataFrame do wykresów
    df_combined = pd.concat([df_train, df_test], ignore_index=True)

    # Wybierzmy tylko kolumny numeryczne
    numeric_cols = df_train.select_dtypes(include=[np.number]).columns.tolist()
    
    # Usuwamy kolumny nie-cechowe
    for col in ["actual_packets_in_flow", "agg_value", "flow_id", "session_id"]:
        if col in numeric_cols:
            numeric_cols.remove(col)

    # Wyliczamy, które cechy zmieniły się najmocniej (np. gdzie wariancja/średnia najbardziej się rozjechała)
    # Prosta heurystyka: różnica średnich podzielona przez odchylenie
    drifts = {}
    for col in numeric_cols:
        mean_train = df_train[col].mean()
        mean_test = df_test[col].mean()
        std = df_combined[col].std()
        if std > 0:
            drift = abs(mean_train - mean_test) / std
            drifts[col] = drift

    # Bierzemy TOP X cech, które NAJBARDZIEJ się różnią między nagraniami
    top_features = sorted(drifts, key=drifts.get, reverse=True)[:top_features_to_plot]

    print(f"[!] Znaleziono cechy z największym odchyleniem (Data Drift) dla {app_name}:")
    for f in top_features:
        print(f" - {f} (wskaźnik odchylenia: {drifts[f]:.2f})")

    # Tworzymy wykresy (Boxploty - świetne do pokazywania różnic)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f"Przesunięcie danych (Data Drift) dla: {app_name}", fontsize=16)

    axes = axes.flatten()
    for i, feature in enumerate(top_features):
        sns.boxplot(x="Zbiór", y=feature, data=df_combined, ax=axes[i], palette="Set2")
        axes[i].set_title(feature)
        axes[i].set_ylabel("")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    compare_datasets(
        csv_path_train="data/csv/train_data.csv",
        csv_path_test="data/csv/test_data_failed.csv",
        app_name="discord"
    )