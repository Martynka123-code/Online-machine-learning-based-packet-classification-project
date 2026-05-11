from sniffing.sniffer_training import SnifferTraining
# --- NEW IMPORTS FOR ONLINE PIPELINE ---
from sniffing.sniffer_online import SnifferOnline
from models.rf_online import RandomForestOnline
from models.cnn_online import CNNOnline
from ui.interface import UIInterface

def run_online_pipeline():
    """Main function controlling the live classification pipeline."""
    print("\n" + "=" * 50)
    print(" STARTING ONLINE PIPELINE ")
    print("=" * 50)

    # 1. Select RF model (using the first granularity from config as default, e.g., 50)
    gran = GRANULARITIES[0] if GRANULARITIES else 50
    rf_model_path = os.path.join(MODELS_DIR, f"rf_model_{gran}.pkl")
    
    if not os.path.exists(rf_model_path):
        print(f"[!] Error: RF model not found at {rf_model_path}. Train the model first (Option 3).")
        return

    # 2. Initialize models
    rf_classifier = RandomForestOnline(rf_model_path)
    cnn_classifier = CNNOnline("data/saved_models/cnn_model.h5") # Placeholder path

    # 3. Buffer for flows (for RF model)
    active_flows = {} 

    def packet_callback(packet):
        """This function is called for EVERY packet captured by the sniffer."""
        
        # --- CNN PATH (Single packet byte-level analysis) ---
        # TODO: Call the logic formatting the packet to N bytes here
        # cnn_input = format_packet_for_cnn(packet, length=750)
        # cnn_pred = cnn_classifier.classify_stream(cnn_input)
        # print(f"[CNN] Prediction: {cnn_pred}")

        # --- RF PATH (Flow aggregation) ---
        # Use FlowFeatureExtractor logic to assign packet to a flow key (5-tuple)
        temp_extractor = FlowFeatureExtractor(None, None) 
        key = temp_extractor._get_flow_key(packet)

        if key:
            if key not in active_flows:
                active_flows[key] = []
            
            active_flows[key].append(packet)

            # If we collected enough packets for the granularity (e.g., 50)
            if len(active_flows[key]) == gran:
                # Calculate statistical features for this batch of packets
                features = temp_extractor._calculate_features(active_flows[key])
                
                # RF Classification
                rf_pred = rf_classifier.classify_stream(features)
                print(f"[RF] FLOW CLASSIFICATION: {key} -> RESULT: {rf_pred}")
                
                # Clear the buffer for this key to collect the next batch
                active_flows[key] = []

    # 4. Start the Sniffer
    sniffer = SnifferOnline(callback_function=packet_callback)
    try:
        sniffer.start_capture()
    except KeyboardInterrupt:
        sniffer.stop_capture()
        print("\n[*] Returning to main menu.")

def menu():
    """Main interactive menu — entry point for all pipeline stages."""
    while True:
        print("\n" + "=" * 50)
        print(" ML NETWORK TRAFFIC CLASSIFIER - MAIN MENU")
        print("=" * 50)
        print("1. Collect training data (Sniffer per process -> PCAP)")
        print("2. Extract features from PCAP files (PCAP -> aggregated CSV)")
        print("3. Train model: Random Forest (Granularity range)")
        print("4. Run online classification (RF & CNN Pipeline)")
        print("5. [Not implemented] Launch GUI")
        print("0. Exit")

        choice = input("\nSelect an option: ")

        if choice == '1':
            app = input("Enter process name (e.g. spotify, firefox): ")
            print(f"[*] Starting sniffer for {app}... ([IN PROGRESS] sniffing/sniffer_training.py)")
            # sniffer = SnifferTraining(app)
            # sniffer.start()

        elif choice == '2':
            print(f"\nAvailable files in {DATA_RAW_DIR}:")
            try:
                files = os.listdir(DATA_RAW_DIR)
                if not files:
                    print("No files found. Use option 1 or place pcap files manually.")
                for f in files:
                    print(f" - {f}")
            except FileNotFoundError:
                print(f"[!] Directory {DATA_RAW_DIR} was not found. Please rerun the program to create it.")
                continue

            pcap_file = input("\nEnter pcap filename to process: ")
            label = input("Please enter label for this traffic class (e.g. Spotify, YouTube): ")
            pcap_path = os.path.join(DATA_RAW_DIR, pcap_file)

            if os.path.exists(pcap_path):
                print(f"\n[*] Starting feature extraction for granularities: {GRANULARITIES}")

                # Create a separate CSV file for each granularity value
                for gran in GRANULARITIES:
                    output_csv = os.path.join(DATA_CSV_DIR, f"rf_dataset_{gran}.csv")
                    extractor = FlowFeatureExtractor(pcap_path, label, granularity=gran)
                    extractor.process_and_save(output_csv)
                print("\n[+] Feature extraction completed successfully! ")
            else:
                print(f"[!] File not found: {pcap_path}")

        elif choice == '3':
            print("\n[*] Starting training sweep across all granularities...")

            results = {}

            for gran in GRANULARITIES:
                csv_path = os.path.join(DATA_CSV_DIR, f"rf_dataset_{gran}.csv")

                if os.path.exists(csv_path):
                    print(f"\n" + "-" * 40)
                    print(f" TRAINING GRANULARITY: {gran} PACKETS ")
                    print("-" * 40)

                    trainer = RandomForestTrainer(csv_path)

                    accuracy = trainer.train_and_evaluate()

                    if accuracy is not None:
                        results[gran] = accuracy

                    # Save a separate model file for each granularity (e.g. rf_model_50.pkl)
                    model_save_path = os.path.join(MODELS_DIR, f"rf_model_{gran}.pkl")
                    trainer.save_model(model_save_path)
                else:
                    print(f"[!] Dataset not found for granularity {gran}: {csv_path}")

            # Print granularity comparison table
            if results:
                print("\n" + "=" * 50)
                print(" GRANULARITY EXPERIMENT SUMMARY ")
                print("=" * 50)
                for g, acc in results.items():
                    print(f" Granularity {g:3} packets -> Accuracy: {acc * 100:.2f}%")
                best = max(results, key=results.get)
                print(f"\nBest granularity: {best} packets ({results[best] * 100:.2f}%)")
                print("=" * 50)

        elif choice == '4':
            run_online_pipeline()
        
        elif choice == '5':
            print("[Not implemented] Launch GUI")

        elif choice == '0':
            print("Exiting. Goodbye!")
            sys.exit(0)

        else:
            print("Unknown option. Please enter a valid number.")


if __name__ == "__main__":
    menu()