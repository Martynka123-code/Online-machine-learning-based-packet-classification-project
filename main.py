import os
import sys

# Importy konfiguracji
from config import DATA_RAW_DIR, DATA_CSV_DIR, MODELS_DIR, GRANULARITIES

# Importy rzeczywistych modułów
from preprocessing.feature_extractor import FlowFeatureExtractor
from models.rf_trainer import RandomForestTrainer

# Importy modułów "Na przyszłość" (zaślepki)
from sniffing.sniffer_training import SnifferTraining
from sniffing.sniffer_online import SnifferOnline
from models.rf_online import RandomForestOnline
from models.cnn_trainer import CNNTrainer
from ui.interface import UIInterface


def menu():
    while True:
        print("\n" + "=" * 50)
        print(" ML NETWORK TRAFFIC CLASSIFIER - PANEL GŁÓWNY")
        print("=" * 50)
        print("1. Zbieraj dane uczące (Sniffer per process -> PCAP)")
        print("2. Wylicz cechy z plików PCAP (Agregacja do CSV)")
        print("3. Ucz model: Random Forest (Test granularności)")
        print("4. [W budowie] Uruchom klasyfikację Online (RF)")
        print("5. [W budowie] Uruchom GUI")
        print("0. Wyjście")

        choice = input("\nWybierz opcję: ")

        if choice == '1':
            app = input("Podaj nazwę procesu (np. spotify, firefox): ")
            print(f"[*] Uruchamianie sniffera dla {app}... (Zaimplementuj kod w sniffing/sniffer_training.py)")
            # sniffer = SnifferTraining(app)
            # sniffer.start()

        elif choice == '2':
            print(f"\nDostępne pliki w {DATA_RAW_DIR}:")
            try:
                files = os.listdir(DATA_RAW_DIR)
                if not files:
                    print("Brak plików. Najpierw użyj opcji 1 lub wrzuć pliki pcap ręcznie.")
                for f in files:
                    print(f" - {f}")
            except FileNotFoundError:
                print(f"[!] Folder {DATA_RAW_DIR} nie istnieje. Uruchom program ponownie, aby go utworzyć.")
                continue

            pcap_file = input("\nPodaj nazwę pliku pcap do przetworzenia: ")
            label = input("Podaj etykietę dla tych danych (np. Spotify, YouTube): ")
            pcap_path = os.path.join(DATA_RAW_DIR, pcap_file)

            if os.path.exists(pcap_path):
                print(f"\n[*] Rozpoczynam ekstrakcję dla granularności: {GRANULARITIES}")
                # Tworzymy osobny plik CSV dla KAŻDEJ granularności
                for gran in GRANULARITIES:
                    output_csv = os.path.join(DATA_CSV_DIR, f"rf_dataset_{gran}.csv")
                    extractor = FlowFeatureExtractor(pcap_path, label, granularity=gran)
                    extractor.process_and_save(output_csv)
                print("\n[+] Ekstrakcja zakończona pomyślnie!")
            else:
                print(f"[!] Nie znaleziono pliku: {pcap_path}")

        elif choice == '3':
            print("\n[*] Rozpoczynam trening modeli dla różnych granularności...")

            # Słownik do zapisywania wyników dla porównania
            results = {}

            for gran in GRANULARITIES:
                csv_path = os.path.join(DATA_CSV_DIR, f"rf_dataset_{gran}.csv")

                if os.path.exists(csv_path):
                    print(f"\n" + "-" * 40)
                    print(f" TRENOWANIE DLA GRANULARNOŚCI: {gran} PAKIETÓW ")
                    print("-" * 40)

                    trainer = RandomForestTrainer(csv_path)

                    # Trenujemy i pobieramy skuteczność (accuracy)
                    accuracy = trainer.train_and_evaluate()

                    if accuracy is not None:
                        results[gran] = accuracy

                    # Zapisujemy osobny model dla danej granularności (np. rf_model_50.pkl)
                    model_save_path = os.path.join(MODELS_DIR, f"rf_model_{gran}.pkl")
                    trainer.save_model(model_save_path)
                else:
                    print(f"[!] Brak pliku datasetu dla granularności {gran}: {csv_path}")

            # PODSUMOWANIE (Gotowe do raportu)
            if results:
                print("\n" + "=" * 50)
                print(" PODSUMOWANIE EKSPERYMENTU Z GRANULARNOŚCIĄ")
                print("=" * 50)
                for g, acc in results.items():
                    print(f"Granularność {g:3} pakietów -> Skuteczność: {acc * 100:.2f}%")
                print("=" * 50)

        elif choice == '0':
            print("Zamykanie programu. Do zobaczenia!")
            sys.exit(0)

        else:
            print("Nieznana opcja lub moduł w budowie. Wybierz poprawny numer.")


if __name__ == "__main__":
    menu()