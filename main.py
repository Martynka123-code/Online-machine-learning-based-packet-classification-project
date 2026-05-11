import os
import sys
from config import DATA_RAW_DIR, DATA_CSV_DIR, MODELS_DIR, GRANULARITIES
from preprocessing.feature_extractor import FlowFeatureExtractor
from models.rf_trainer import RandomForestTrainer
from models.rf_online import RandomForestOnline
from sniffing.sniffer_online import SnifferOnline
from visualization.rf_visualizer import plot_granularity_comparison

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
    gran = int(choice) if choice else GRANULARITIES[0]

    model_path = os.path.join(MODELS_DIR, f"rf_model_{gran}.pkl")
    if not os.path.exists(model_path):
        print(f"[!] Model not found: {model_path}. Please train the model first.")
        return

    rf_classifier = RandomForestOnline(model_path)
    # Temporary extractor for real-time feature calculation
    extractor = FlowFeatureExtractor(None, None, granularity=gran)
    active_flows = {}

    def packet_processing_logic(packet):
        """Callback function for the sniffer to process packets in real-time."""
        key = extractor._get_flow_key(packet)
        if key:
            if key not in active_flows:
                active_flows[key] = []
            active_flows[key].append(packet)

            if len(active_flows[key]) == gran:
                features = extractor._calculate_features(active_flows[key])
                prediction = rf_classifier.classify_stream(features)
                print(f"[LIVE] Flow: {key} | Application: {prediction}")
                active_flows[key] = []

    sniffer = SnifferOnline(packet_processing_logic)
    try:
        sniffer.start_capture()
    except KeyboardInterrupt:
        sniffer.stop_capture()
        print("\n[*] Terminating online mode and returning to menu.")

def main_menu():
    """Primary application entry point and CLI menu router."""
    while True:
        print("\n" + "═"*50)
        print(" OMLBPC PROJECT - CORE SYSTEM INTERFACE ")
        print("═"*50)
        print(" 1. Collect Training Data (Sniffer to PCAP)")
        print(" 2. Feature Extraction    (PCAP to CSV)")
        print(" 3. Train Models          (RF Training Sweep)")
        print(" 4. Online Mode           (Real-time Classification)")
        print(" 0. Exit System")

        cmd = input("\nExecute option: ").strip()

        if cmd == '1': print("[*] Call data collection module...")
        elif cmd == '2': print("[*] Call extraction module...")
        elif cmd == '3': menu_train_models()
        elif cmd == '4': menu_online_classification()
        elif cmd == '0':
            print("System shutdown. Goodbye.")
            sys.exit(0)
        else:
            print("[!] Invalid command.")

if __name__ == "__main__":
    main_menu()