"""
batch_evaluator.py
------------------
Menu CLI do ewaluacji wielu plików PCAP naraz.
Wyniki agreguje i przekazuje do report_generator.
"""

import os

from config import DATA_TEST_DIR, MODELS_DIR, DATA_CNN_DIR
from evaluation.offline_evaluator import evaluate_pcap
from evaluation.report_generator import print_single_report, print_batch_report


# ---------------------------------------------------------------------------
# Pomocnicze funkcje wyboru
# ---------------------------------------------------------------------------

def _pick_pcap_files():
    """Wyświetla dostępne pliki PCAP i pozwala wybrać jeden lub wiele."""
    files = sorted([f for f in os.listdir(DATA_TEST_DIR) if f.endswith(".pcap")])
    if not files:
        print(f"[!] Brak plików .pcap w {DATA_TEST_DIR}")
        return []

    print("\n  Dostępne pliki PCAP:")
    for idx, f in enumerate(files, 1):
        print(f"    {idx:>2}. {f}")

    raw = input(
        "\n  Wybierz numery plików (np. 1 3 5) lub 'all' dla wszystkich: "
    ).strip().lower()

    if raw == "all":
        return [os.path.join(DATA_TEST_DIR, f) for f in files]

    chosen = []
    for token in raw.split():
        try:
            idx = int(token) - 1
            chosen.append(os.path.join(DATA_TEST_DIR, files[idx]))
        except (ValueError, IndexError):
            print(f"  [!] Pominięto nieprawidłowy wybór: '{token}'")

    return chosen


def _pick_model():
    """Zwraca (model_type, model_params) na podstawie wyboru użytkownika."""
    model_type = input("\n  Wybierz model [rf / cnn]: ").strip().lower()
    if model_type not in ("rf", "cnn"):
        print("  [!] Nieznany typ modelu.")
        return None, None

    model_params = {}

    if model_type == "rf":
        from config import PACKET_GRANULARITIES, TIME_WINDOWS

        agg_mode = input("  Tryb agregacji [packet / time]: ").strip().lower()
        if agg_mode not in ("packet", "time"):
            print("  [!] Nieprawidłowy tryb. Używam 'packet'.")
            agg_mode = "packet"

        if agg_mode == "packet":
            print(f"  Granularności pakietowe: {PACKET_GRANULARITIES}")
            choice = input(f"  Wybierz wartość (domyślnie {PACKET_GRANULARITIES[0]}): ").strip()
            agg_value = int(choice) if choice else PACKET_GRANULARITIES[0]
        else:
            print(f"  Okna czasowe (s): {TIME_WINDOWS}")
            choice = input(f"  Wybierz wartość (domyślnie {TIME_WINDOWS[0]}): ").strip()
            agg_value = float(choice) if choice else TIME_WINDOWS[0]

        model_params = {"agg_mode": agg_mode, "agg_value": agg_value}

    return model_type, model_params


def _assign_labels(pcap_paths):
    """
    Pozwala przypisać prawdziwą klasę do każdego pliku.
    Jeśli nazwa pliku zaczyna się od nazwy klasy (np. spotify_capture.pcap),
    proponuje ją jako domyślną.
    """
    label_map = {}
    print()
    for path in pcap_paths:
        fname = os.path.basename(path)
        # Próba zgadnięcia klasy z nazwy pliku
        suggested = fname.split("_")[0].split(".")[0].lower()
        raw = input(
            f"  Prawdziwa klasa dla '{fname}' [Enter = '{suggested}']: "
        ).strip().lower()
        label_map[path] = raw if raw else suggested
    return label_map


# ---------------------------------------------------------------------------
# Główne menu
# ---------------------------------------------------------------------------

def menu_batch_evaluation():
    """Główny punkt wejścia wywoływany z main.py."""
    print("\n" + "=" * 60)
    print(" OFFLINE BATCH EVALUATION ".center(60))
    print("=" * 60)

    # 1. Wybór plików
    pcap_paths = _pick_pcap_files()
    if not pcap_paths:
        return

    print(f"\n  Wybrano {len(pcap_paths)} plik(ów).")

    # 2. Przypisanie etykiet
    label_map = _assign_labels(pcap_paths)

    # 3. Wybór modelu
    model_type, model_params = _pick_model()
    if model_type is None:
        return

    # 4. Ewaluacja
    all_stats = []
    for path in pcap_paths:
        true_label = label_map[path]
        print(f"\n  >>> Plik: {os.path.basename(path)}  |  Klasa: {true_label.upper()}")
        try:
            stats = evaluate_pcap(path, true_label, model_type, model_params)
            all_stats.append(stats)

            # Pojedynczy raport po każdym pliku (opcjonalny)
            if len(pcap_paths) > 1:
                show = input("  Pokazać szczegółowy raport tego pliku? [t/N]: ").strip().lower()
                if show == "t":
                    print_single_report(stats)
            else:
                print_single_report(stats)

        except (FileNotFoundError, RuntimeError) as e:
            print(f"  [!] Błąd: {e}")
            all_stats.append(None)

    # 5. Raport zbiorczy (gdy więcej niż 1 plik)
    if len(pcap_paths) > 1:
        print_batch_report(all_stats)