# config.py
import os

# Ścieżki do folderów
DATA_RAW_DIR = "data/raw_pcaps/"
DATA_CSV_DIR = "data/processed_csv/"
MODELS_DIR = "data/saved_models/"

# Tworzenie folderów jeśli nie istnieją
for d in [DATA_RAW_DIR, DATA_CSV_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

# Ustawienia ekstrakcji cech
GRANULARITIES = [50, 100, 150]