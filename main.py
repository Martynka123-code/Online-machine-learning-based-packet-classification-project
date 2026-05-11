import os
import sys
import threading

from config import DATA_RAW_DIR, DATA_CSV_DIR, MODELS_DIR, GRANULARITIES
from preprocessing.feature_extractor import FlowFeatureExtractor
from models.rf_trainer import RandomForestTrainer
from models.rf_online import RandomForestOnline
from sniffing.sniffer_online import SnifferOnline
from sniffing.sniffer_training import SnifferTraining
from visualization.rf_visualizer import plot_granularity_comparison

def _ensure_dirs():
    """Creates necessary directories if they do not exist."""
    for d in [DATA_RAW_DIR, DATA_CSV_DIR, MODELS_DIR, "reports"]:
        os.makedirs(d, exist_ok=True)

def menu_collect_data():
    """
    Handles capturing raw network traffic and saving it to a PCAP file.
    """
    print("\n" + "-"*40)
    app = input("Enter process/app name (e.g. spotify, firefox): ").strip()
    print(f"[*] Starting offline sniffer for '{app}'...")
    sniffer = SnifferTraining(app)
    sniffer.start()

def menu_extract_features():
    """
    Reads a PCAP file, extracts statistical features for different granularities,
    and saves them into aggregated CSV datasets.
    """
    print("\n" + "-"*40)
    try:
        files = os.listdir(DATA_RAW_DIR)
        pcap_files = [f for f in files if f.endswith('.pcap')]
    except FileNotFoundError:
        pcap_files = []

    if not pcap_files:
        print(f"[!] No .pcap files found in {DATA_RAW_DIR}. Place files there first.")
        return

    print(f"Available files in {DATA_RAW_DIR}:")
    for f in pcap_files:
        print(f"  - {f}")

    pcap_file = input("\nEnter PCAP filename to process: ").strip()
    label = input("Traffic class label (e.g. Spotify, YouTube, Background): ").strip()
    pcap_path = os.path.join(DATA_RAW_DIR, pcap_file)

    if not os.path.exists(pcap_path):
        print(f"[!] File not found: {pcap_path}")
        return

    print(f"\n[*] Extracting features for granularities: {GRANULARITIES}")
    for gran in GRANULARITIES:
        output_csv = os.path.join(DATA_CSV_DIR, f"rf_dataset_{gran}.csv")
        extractor = FlowFeatureExtractor(pcap_path, label, granularity=gran)
        extractor.process_and_save(output_csv)

    print("\n[+] Feature extraction completed successfully!")

def menu_train_models():
    """
    Orchestrates the training sweep across all configured granularities.
    Saves trained models and generates a performance summary plot.
    """
    print("\n[*] Initializing training sweep across all granularities...")
    training_results = {}

    for gran in GRANULARITIES:
        csv_path = os.path.join(DATA_CSV_DIR, f"rf_dataset_{gran}.csv")
        if not os.path.exists(csv_path):
            print(f"[!] Dataset missing for granularity {gran}: {csv_path}")
            continue

        print(f"\n--- Training for Granularity: {gran} packets ---")
        trainer = RandomForestTrainer(csv_path)
        accuracy = trainer.train_and_evaluate()
        
        if accuracy:
            training_results[gran] = accuracy
            model_file = f"rf_model_{gran}.pkl"
            trainer.save_model(os.path.join(MODELS_DIR, model_file))

    if training_results:
        plot_granularity_comparison(training_results)

def menu_online_classification():
    """
    Launches the real-time classification pipeline.
    Allows user to select a pre-trained model and starts the network sniffer.
    """
    print("\n" + "="*50)
    print(" LIVE NETWORK TRAFFIC CLASSIFICATION PIPELINE ")
    print("="*50)

    print(f"Available granularities: {GRANULARITIES}")
    choice = input(f"Select granularity to use (default {GRANULARITIES[0]}): ").strip()
    
    try:
        gran = int(choice) if choice else GRANULARITIES[0]
        if gran not in GRANULARITIES:
            gran = GRANULARITIES[0]
    except ValueError:
        gran = GRANULARITIES[0]

    model_path = os.path.join(MODELS_DIR, f"rf_model_{gran}.pkl")
    if not os.path.exists(model_path):
        print(f"[!] Model not found: {model_path}. Please train the model first (Option 3).")
        return

    rf_classifier = RandomForestOnline(model_path)
    extractor = FlowFeatureExtractor(None, None, granularity=gran)
    
    active_flows = {}
    
    flow_lock = threading.Lock() 

    def packet_processing_logic(packet):
        """Callback function for the sniffer to process packets in real-time."""
        key = extractor._get_flow_key(packet)
        
        if key is None:
            return  

        with flow_lock:
            features = extractor._calculate_features(active_flows[key][:gran])
            del active_flows[key]

        prediction = rf_classifier.classify_stream(features)
        print(f"[LIVE] Flow: {key[0]}:{key[2]} <-> {key[1]}:{key[3]} | App: {prediction}")

    sniffer = SnifferOnline(packet_processing_logic)
    try:
        sniffer.start_capture()
    except KeyboardInterrupt:
        sniffer.stop_capture()
        print("\n[*] Terminating online mode and returning to menu.")

def main_menu():
    """Primary application entry point and CLI menu router."""
    _ensure_dirs()
    
    while True:
        print("\n" + "═"*55)
        print(" UM PROJECT - CORE SYSTEM INTERFACE ")
        print("═"*55)
        print(" 1. Collect Training Data (Sniffer -> PCAP)")
        print(" 2. Feature Extraction    (PCAP -> CSV)")
        print(" 3. Train Models          (RF Training Sweep)")
        print(" 4. Online Mode           (Real-time Classification)")
        print(" 0. Exit System")

        cmd = input("\nExecute option: ").strip()

        if cmd == '1': menu_collect_data()
        elif cmd == '2': menu_extract_features()
        elif cmd == '3': menu_train_models()
        elif cmd == '4': menu_online_classification()
        elif cmd == '0':
            print("System shutdown. Goodbye.")
            sys.exit(0)
        else:
            print("[!] Invalid command. Please select a valid number.")

if __name__ == "__main__":
    main_menu()