import os

# Paths to directories
DATA_RAW_DIR = "data/raw_pcaps/"
DATA_CSV_DIR = "data/processed_csv/"
MODELS_DIR = "data/saved_models/"
DATA_CNN_DIR = "data/cnn_datasets/"
DATA_TEST_DIR = "data/test_pcaps/"

# Creating directories if they do not exist
for d in [DATA_RAW_DIR, DATA_CSV_DIR, MODELS_DIR, DATA_CNN_DIR, DATA_TEST_DIR]:
    os.makedirs(d, exist_ok=True)

# Feature extraction settings
# POPRAWKA: była zduplikowana lista GRANULARITIES == PACKET_GRANULARITIES
# Zostawiamy jedną nazwę PACKET_GRANULARITIES, GRANULARITIES to alias dla kompatybilności
PACKET_GRANULARITIES = [10, 25, 50, 100, 150]
GRANULARITIES = PACKET_GRANULARITIES  # alias — nie duplikuj wartości

TIME_WINDOWS = [0.5, 1.0, 2.0, 4.0]