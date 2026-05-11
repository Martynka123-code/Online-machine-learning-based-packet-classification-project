# config.py
import os

# Paths to directories
DATA_RAW_DIR = "data/raw_pcaps/"
DATA_CSV_DIR = "data/processed_csv/"
MODELS_DIR = "data/saved_models/"

# Creating directories if they do not exist
for d in [DATA_RAW_DIR, DATA_CSV_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

# Feature extraction settings
GRANULARITIES = [50, 100, 150]