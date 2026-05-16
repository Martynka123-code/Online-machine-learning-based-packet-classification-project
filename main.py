import os
import time
import threading
import queue
import sys
from scapy.all import sniff 

# Import configuration and custom modules
from config import DATA_RAW_DIR, DATA_CSV_DIR, MODELS_DIR, GRANULARITIES
from sniffing.sniffer_training import SnifferTraining
from sniffing.sniffer_online import SnifferOnline
from preprocessing.feature_extractor import FlowFeatureExtractor
from models.rf_trainer import RandomForestTrainer
from models.rf_online import RandomForestOnline
from visualization.rf_visualizer import plot_granularity_comparison


def menu_collect_data():
    """Option 1: Collect raw traffic using the process-to-port sniffer."""
    print("\n--- Data Collection Mode ---")
    app_name = input("Enter application name to monitor (e.g., spotify): ").strip().lower()
    if not app_name: return
    
    sniffer = SnifferTraining(target_apps=[app_name])
    sniffer.start()

def menu_extract_features():
    """Option 2: Convert PCAP files to CSV datasets for multiple granularities."""
    print("\n--- Feature Extraction (PCAP to CSV) ---")
    files = [f for f in os.listdir(DATA_RAW_DIR) if f.endswith('.pcap')]
    if not files:
        print(f"[!] No PCAP files found in {DATA_RAW_DIR}")
        return

    print("\nAvailable PCAP files:")
    for f in files: print(f"  - {f}")
    
    pcap_file = input("\nEnter filename to process: ").strip()
    label = input("Enter traffic label (e.g., Spotify): ").strip()
    pcap_path = os.path.join(DATA_RAW_DIR, pcap_file)

    if not os.path.exists(pcap_path):
        print("[!] File not found.")
        return

    print(f"[*] Extracting features for granularities: {GRANULARITIES}")
    for gran in GRANULARITIES:
        output_csv = os.path.join(DATA_CSV_DIR, f"rf_dataset_{gran}.csv")
        extractor = FlowFeatureExtractor(pcap_path, label, granularity=gran)
        extractor.process_and_save(output_csv)

def menu_validate_datasets():
    """Option 3: Quality check for generated CSV datasets (from test2)."""
    print("\n--- Feature Validation Report ---")
    csv_files = [f for f in os.listdir(DATA_CSV_DIR) if f.endswith('.csv')]
    if not csv_files:
        print("[!] No CSV datasets found. Run extraction first.")
        return

    for csv_file in csv_files:
        path = os.path.join(DATA_CSV_DIR, csv_file)
        print(f"\nValidating: {csv_file}")
        trainer = RandomForestTrainer(path)
        trainer.validate_features()

def menu_train_models():
    """Option 4: Train RF models with advanced analytics and visualization."""
    print("\n--- Model Training & Advanced Analytics ---")
    results = {}
    
    for gran in GRANULARITIES:
        csv_path = os.path.join(DATA_CSV_DIR, f"rf_dataset_{gran}.csv")
        if os.path.exists(csv_path):
            print(f"\n>>> Training for Granularity: {gran} packets")
            # The new RandomForestTrainer will auto-generate plots during train_and_evaluate()
            trainer = RandomForestTrainer(csv_path, reports_dir="reports")
            acc = trainer.train_and_evaluate()
            
            if acc:
                results[gran] = acc
                model_path = os.path.join(MODELS_DIR, f"rf_model_{gran}.pkl")
                trainer.save_model(model_path)
    
    if results:
        print("\n--- Training Summary ---")
        for g, a in sorted(results.items()):
            print(f"Granularity {g:3}: Accuracy {a:.2%}")
        plot_granularity_comparison(results)

def menu_online_mode():
    """Option 5: Live classification with asynchronous packet processing buffer."""
    print("\n--- Online Classification (Real-time) ---")
    
    agg_mode = input("Select aggregation mode [packet / time]: ").strip().lower()
    if agg_mode not in ['packet', 'time']:
        print("[!] Invalid mode. Defaulting to 'packet'.")
        agg_mode = 'packet'
        
    if agg_mode == 'packet':
        from config import PACKET_GRANULARITIES
        print(f"Available packet granularities: {PACKET_GRANULARITIES}")
        choice = input(f"Select value (default {PACKET_GRANULARITIES[0]}): ").strip()
        agg_value = int(choice) if choice else PACKET_GRANULARITIES[0]
        model_name = f"rf_model_{agg_value}.pkl" 
    else:
        from config import TIME_WINDOWS
        print(f"Available time windows (seconds): {TIME_WINDOWS}")
        choice = input(f"Select value (default {TIME_WINDOWS[0]}): ").strip()
        agg_value = float(choice) if choice else TIME_WINDOWS[0]
        model_name = f"rf_model_time_{agg_value}.pkl" 

    model_path = os.path.join(MODELS_DIR, model_name)
    if not os.path.exists(model_path):
        print(f"[!] Model '{model_name}' not found. Train it first (Option 4).")
        return

    classifier = RandomForestOnline(model_path)
    extractor = FlowFeatureExtractor(pcap_path=None, label=None, agg_mode=agg_mode, agg_value=agg_value)
    sniffer = SnifferOnline()
    active_flows = {}
    
    def packet_processor():
        """Background worker thread to handle packet buffering and classification."""
        print(f"[*] Worker thread started: processing packets in '{agg_mode}' mode (val: {agg_value})...")
        FLOW_TIMEOUT = 120.0 
        
        while True:
            try:
                packet = sniffer.packet_queue.get(timeout=1.0) 
                if packet is None: break
                    
                key = extractor._get_flow_key(packet)
                if key is None: continue

                if key not in active_flows:
                    active_flows[key] = {"packets": [], "last_seen": time.time()}
                
                active_flows[key]["packets"].append(packet)
                active_flows[key]["last_seen"] = time.time()

                flow_packets = active_flows[key]["packets"]
                flush_flow = False

                if agg_mode == "packet":
                    if len(flow_packets) >= agg_value:
                        flush_flow = True
                elif agg_mode == "time":
                    time_diff = float(flow_packets[-1].time - flow_packets[0].time)
                    if time_diff >= agg_value:
                        flush_flow = True

                if flush_flow:
                    features = extractor._calculate_features(flow_packets)
                    del active_flows[key] 

                    result = classifier.classify_stream(features)
                    
                    pred = result.get("prediction", "UNKNOWN")
                    conf = result.get("confidence", 0.0)
                    print(f"[LIVE] Flow {key[4]}: {key[0]}:{key[2]} <-> {key[1]}:{key[3]} | App: {pred} ({conf*100:.1f}%)")

                sniffer.packet_queue.task_done()

            except queue.Empty:
                # Cleanup stale flows that haven't seen packets for a while
                current_time = time.time()
                stale_keys = [k for k, v in active_flows.items() if current_time - v["last_seen"] > FLOW_TIMEOUT]
                for k in stale_keys: del active_flows[k]
            except Exception as e:
                print(f"[ERROR] {e}")

    print(f"[*] Loaded model: {model_path}")
    processor_thread = threading.Thread(target=packet_processor, daemon=True)
    processor_thread.start()

    try:
        sniffer.start_capture()
    except KeyboardInterrupt:
        print("\n[*] Stopping capture...")
        sniffer.stop_capture()
        processor_thread.join(timeout=1.0)

def main():
    """Main application loop."""
    while True:
        print("\n" + "="*55)
        print(" NETWORK TRAFFIC CLASSIFIER - INTEGRATED SYSTEM ")
        print("="*55)
        print(" 1. Collect Training Data (Sniffer)")
        print(" 2. Extract Features (PCAP to CSV)")
        print(" 3. Validate Datasets (Quality Report)")
        print(" 4. Train Models & Analytics (RF)")
        print(" 5. Run Online Classification (Live)")
        print(" 0. Exit")
        
        cmd = input("\nSelect option: ").strip()
        if cmd == '1': menu_collect_data()
        elif cmd == '2': menu_extract_features()
        elif cmd == '3': menu_validate_datasets()
        elif cmd == '4': menu_train_models()
        elif cmd == '5': menu_online_mode()
        elif cmd == '0': break

if __name__ == "__main__":
    # Ensure necessary directories exist
    for d in [DATA_RAW_DIR, DATA_CSV_DIR, MODELS_DIR, "reports"]:
        os.makedirs(d, exist_ok=True)
    main()