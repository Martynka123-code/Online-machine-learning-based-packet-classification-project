import os
import sys

from config import DATA_RAW_DIR, DATA_CSV_DIR, MODELS_DIR, GRANULARITIES

from preprocessing.feature_extractor import FlowFeatureExtractor
from models.rf_trainer import RandomForestTrainer

from sniffing.sniffer_training import SnifferTraining
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
    gran = GRANULARITIES[0] if GRANULARITIES else 150
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

def menu_online():
    """Main function controlling the live classification pipeline."""
    print("\n" + "=" * 50)
    print(" STARTING ONLINE PIPELINE ")
    print("=" * 50)

    # 1. Wybór Granulacji przez użytkownika
    print(f"Available models for granularities: {GRANULARITIES}")
    choice = input(f"Select granularity to use (default {GRANULARITIES[0]}): ").strip()
    
    try:
        gran = int(choice) if choice else GRANULARITIES[0]
        if gran not in GRANULARITIES:
            print(f"[!] Granularity {gran} is not valid. Reverting to {GRANULARITIES[0]}.")
            gran = GRANULARITIES[0]
    except ValueError:
        print(f"[!] Invalid input. Reverting to {GRANULARITIES[0]}.")
        gran = GRANULARITIES[0]

    rf_model_path = os.path.join(MODELS_DIR, f"rf_model_{gran}.pkl")
    
    if not os.path.exists(rf_model_path):
        print(f"[!] Error: RF model not found at {rf_model_path}. Train the model first (Option 4).")
        return

    # 2. Initialize models
    rf_classifier = RandomForestOnline(rf_model_path)
    cnn_classifier = CNNOnline("data/saved_models/cnn_model.h5") # Placeholder path

    # 3. Buffer for flows (for RF model)
    active_flows = {} 

    def packet_callback(packet):
        # --- RF PATH (Flow aggregation) ---
        temp_extractor = FlowFeatureExtractor(None, None) 
        key = temp_extractor._get_flow_key(packet)

        if key:
            if key not in active_flows:
                active_flows[key] = []
            
            active_flows[key].append(packet)

            if len(active_flows[key]) == gran:
                features = temp_extractor._calculate_features(active_flows[key])
                rf_pred = rf_classifier.classify_stream(features)
                print(f"[RF] FLOW CLASSIFICATION: {key} -> RESULT: {rf_pred}")
                active_flows[key] = []

    # 4. Start the Sniffer
    sniffer = SnifferOnline(callback_function=packet_callback)
    try:
        sniffer.start_capture()
    except KeyboardInterrupt:
        sniffer.stop_capture()
        print("\n[*] Returning to main menu.")


if __name__ == "__main__":
    menu_online()