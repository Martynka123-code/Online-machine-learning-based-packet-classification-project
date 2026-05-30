import os

# Paths to directories
DATA_RAW_DIR = "data/raw_pcaps/"
DATA_CSV_DIR = "data/processed_csv/"
MODELS_DIR = "data/saved_models/"
DATA_CNN_DIR = "data/cnn_datasets/"

# Creating directories if they do not exist
for d in [DATA_RAW_DIR, DATA_CSV_DIR, MODELS_DIR, DATA_CNN_DIR]:
    os.makedirs(d, exist_ok=True)

# Feature extraction settings
GRANULARITIES = [10, 25, 50, 100, 150]
PACKET_GRANULARITIES = [10, 25, 50, 100, 150]

TIME_WINDOWS = [0.5, 1.0, 2.0, 4.0]