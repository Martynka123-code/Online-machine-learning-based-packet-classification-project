import os
import threading
from scapy.all import sniff 
from config import DATA_RAW_DIR, DATA_CSV_DIR, MODELS_DIR, GRANULARITIES

from sniffing.sniffer_training import SnifferTraining
from sniffing.sniffer_online import SnifferOnline
from preprocessing.feature_extractor import FlowFeatureExtractor
from models.rf_trainer import RandomForestTrainer
from models.rf_online import RandomForestOnline
from visualization.rf_visualizer import plot_granularity_comparison

def menu_collect_data():
    """Option 1: Collect raw traffic using the advanced process-to-port sniffer."""
    app_name = input("Enter application name to monitor (e.g., spotify): ").strip().lower()
    if not app_name: return
    
    sniffer = SnifferTraining(target_apps=[app_name])
    sniffer.start()

def menu_extract_features():
    """Option 2: Convert PCAP files to CSV datasets."""
    files = [f for f in os.listdir(DATA_RAW_DIR) if f.endswith('.pcap')]
    if not files:
        print(f"[!] No PCAP files found in {DATA_RAW_DIR}")
        return

    print("\nAvailable files:")
    for f in files: print(f"  - {f}")
    
    pcap_file = input("\nEnter filename to process: ").strip()
    label = input("Enter traffic label (e.g., Spotify): ").strip()
    pcap_path = os.path.join(DATA_RAW_DIR, pcap_file)

    print(f"[*] Extracting features for granularities: {GRANULARITIES}")
    for gran in GRANULARITIES:
        output_csv = os.path.join(DATA_CSV_DIR, f"rf_dataset_{gran}.csv")
        extractor = FlowFeatureExtractor(pcap_path, label, granularity=gran)
        extractor.process_and_save(output_csv)

def menu_train_models():
    """Option 3: Train Random Forest models and generate reports."""
    results = {}
    for gran in GRANULARITIES:
        csv_path = os.path.join(DATA_CSV_DIR, f"rf_dataset_{gran}.csv")
        if os.path.exists(csv_path):
            trainer = RandomForestTrainer(csv_path)
            acc = trainer.train_and_evaluate()
            if acc:
                results[gran] = acc
                trainer.save_model(os.path.join(MODELS_DIR, f"rf_model_{gran}.pkl"))
    
    if results:
        plot_granularity_comparison(results)

import threading

def menu_online_mode():
    """Option 4: Live classification pipeline with asynchronous buffering."""
    print(f"\nAvailable granularities: {GRANULARITIES}")
    choice = input(f"Select granularity (default {GRANULARITIES[0]}): ").strip()
    gran = int(choice) if choice else GRANULARITIES[0]

    model_path = os.path.join(MODELS_DIR, f"rf_model_{gran}.pkl")
    if not os.path.exists(model_path):
        print("[!] Model file not found. Train it first.")
        return

    classifier = RandomForestOnline(model_path)
    extractor = FlowFeatureExtractor(None, None, granularity=gran)
    
    sniffer = SnifferOnline()
    active_flows = {}
    
    def packet_processor():
        print("[*] Worker thread started: waiting for packets in buffer...")
        while True:
            packet = sniffer.packet_queue.get() 
            
            if packet is None:
                break
                
            try:
                key = extractor._get_flow_key(packet)
                if key is None:
                    continue

                if key not in active_flows:
                    active_flows[key] = []
                active_flows[key].append(packet)

                if len(active_flows[key]) >= gran:
                    features = extractor._calculate_features(active_flows[key][:gran])
                    del active_flows[key] 

                    prediction = classifier.classify_stream(features)
                    print(f"[LIVE] Flow: {key[0]}:{key[2]} <-> {key[1]}:{key[3]} | App: {prediction}")

            except Exception as e:
                print(f"[ERROR in processor] {type(e).__name__}: {e}")
                
            finally:
                sniffer.packet_queue.task_done()


    print(f"[*] Loaded Random Forest model from: {model_path}")
    
    processor_thread = threading.Thread(target=packet_processor, daemon=True)
    processor_thread.start()

    print("[*] Press Ctrl+C to stop.")
    try:
        sniffer.start_capture()
    except KeyboardInterrupt:
        print("\n[*] Stopping capture and returning to menu...")
        sniffer.stop_capture()
        processor_thread.join(timeout=1.0)

def main():
    while True:
        print("\n" + "="*50)
        print(" NETWORK CLASSIFIER SYSTEM ")
        print("="*50)
        print(" 1. Collect Training Data (Advanced Sniffer)")
        print(" 2. Extract Features (PCAP to CSV)")
        print(" 3. Train Models (Random Forest)")
        print(" 4. Run Online Classification")
        print(" 0. Exit")
        
        cmd = input("\nSelect option: ").strip()
        if cmd == '1': menu_collect_data()
        elif cmd == '2': menu_extract_features()
        elif cmd == '3': menu_train_models()
        elif cmd == '4': menu_online_mode()
        elif cmd == '0': break

if __name__ == "__main__":
    main()