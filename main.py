from datetime import datetime
import os
import time
import threading
import torch
import queue
import json
from scapy.all import sniff
import numpy as np

# Import configuration and custom modules
from config import DATA_CNN_DIR, DATA_RAW_DIR, DATA_CSV_DIR, MODELS_DIR, GRANULARITIES
from models.cnn_online import CNNOnline
from preprocessing.cnn_preprocessor import CNNPreprocessor
from sniffing.sniffer_training import SnifferTraining
from sniffing.sniffer_online import SnifferOnline
from preprocessing.feature_extractor import FlowFeatureExtractor
from models.rf_trainer import RandomForestTrainer
from models.rf_online import RandomForestOnline
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import Trainer
from models.cnn_trainer import OptimizedPacketCNN, PacketByteDataset
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
        extractor = FlowFeatureExtractor(pcap_path=pcap_path, label=label, agg_mode='packet', agg_value=gran)
        extractor.process_and_save(output_csv)


def menu_extract_features_cnn():
    """Option 3: Convert ALL available PCAP files into a single master NPZ dataset for CNN."""
    print("\n--- Batch Feature Extraction for CNN (All PCAPs to Single Master NPZ) ---")

    files = [f for f in os.listdir(DATA_RAW_DIR) if f.endswith('.pcap')]
    if not files:
        print(f"[!] No PCAP files found in {DATA_RAW_DIR}")
        return

    # 1. Automatically detect unique applications based on the filename structure (app_date.pcap)
    detected_apps = sorted(list(set([os.path.splitext(f)[0].split('_')[0] for f in files])))

    # 2. Dynamically build a class-to-index mapping dictionary
    class_map = {app_name: idx for idx, app_name in enumerate(detected_apps)}

    print("\n[+] Automatically discovered applications and assigned indices:")
    for app_name, idx in class_map.items():
        print(f"  Class {idx}: {app_name}")

    # Save the class map to a JSON file so the online module knows which index maps to which app
    map_path = os.path.join(DATA_CSV_DIR, "cnn_class_map.json")
    with open(map_path, 'w') as f:
        json.dump(class_map, f, indent=4)
    print(f"[+] Class mapping dictionary saved to: {map_path}")

    # 3. Initialize preprocessor and loop through all PCAP files to accumulate data
    preprocessor = CNNPreprocessor(max_length=1000)
    print("\n[*] Processing PCAP files into a combined memory array...")

    for f in files:
        app_name = os.path.splitext(f)[0].split('_')[0]
        label_idx = class_map[app_name]
        pcap_path = os.path.join(DATA_RAW_DIR, f)

        # This will append tensors and labels internally into preprocessor.data and preprocessor.labels
        preprocessor.process_pcap(pcap_path, label_idx)

    # 4. Save everything into one combined master dataset file
    output_npz = os.path.join(DATA_CNN_DIR, f"cnn_dataset_{datetime.now().strftime('%Y-%m-%d')}_master.npz")
    preprocessor.save_dataset(output_npz)


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

def menu_train_cnn_models():
    """Option 6: Train CNN models using PyTorch Lightning with raw byte datasets."""
    print("\n--- CNN Model Training (PyTorch Lightning) ---")

    files = [f for f in os.listdir(DATA_CNN_DIR) if f.endswith('.npz')]
    if not files:
        print(f"[!] No CNN datasets (.npz) found in {DATA_CNN_DIR}. Run CNN feature extraction first.")
        return

    print("\nAvailable CNN Datasets (.npz):")
    for f in files:
        print(f"  - {f}")

    dataset_file = input("\nEnter dataset filename to train on: ").strip()
    dataset_path = os.path.join(DATA_CNN_DIR, dataset_file)

    if not os.path.exists(dataset_path):
        print("[!] File not found.")
        return

    try:
        full_dataset = PacketByteDataset(dataset_path)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    except Exception as e:
        print(f"[ERROR] Failed to prepare DataLoaders: {e}")
        return

    unique_labels = np.unique(full_dataset.labels)
    output_dim = len(unique_labels)
    print(f"[+] Automatically detected output_dim: {output_dim} target classes.")

    try:
        epochs = int(input("Enter number of training epochs (default 10): ") or 10)
    except ValueError:
        print("[!] Invalid inputs. Numbers must be integers.")
        return

    model = OptimizedPacketCNN(output_dim=output_dim, signal_length=1000)

    from models.cnn_trainer import MetricsCallback
    metrics_cb = MetricsCallback()

    print("\n[*] Starting PyTorch Lightning Training Session...")
    trainer = Trainer(max_epochs=epochs, accelerator="auto", devices=1, callbacks=[metrics_cb])

    try:
        trainer.fit(model, train_loader, val_loader)

        model_base_name = os.path.splitext(dataset_file)[0].replace("cnn_dataset_", "")
        model_save_path = os.path.join(MODELS_DIR, f"cnn_model_{model_base_name}.ckpt")
        trainer.save_checkpoint(model_save_path)
        print(f"\n[+] CNN Training complete! Checkpoint saved to: {model_save_path}")

        # --- Visualizations ---
        from visualization.cnn_visualizer import (
            plot_cnn_training_curves,
            plot_cnn_class_distribution,
            plot_cnn_confusion_matrix,
            plot_cnn_f1_per_class,
        )

        tag = f"cnn_{model_base_name}"

        if metrics_cb.history["train_loss"]:
            plot_cnn_training_curves(
                train_losses=metrics_cb.history["train_loss"],
                val_losses=metrics_cb.history["val_loss"],
                val_accuracies=metrics_cb.history["val_acc"],
                val_f1_scores=metrics_cb.history["val_f1_macro"] or None,
                reports_dir="reports",
                tag=tag,
            )

        class_map_path = os.path.join(DATA_CNN_DIR, "cnn_class_map.json")
        if os.path.exists(class_map_path):
            import json
            with open(class_map_path) as f:
                class_map = json.load(f)

            plot_cnn_class_distribution(
                labels=full_dataset.labels,
                class_map=class_map,
                reports_dir="reports",
                tag=tag,
            )

            # Confusion matrix + per-class F1 on validation split
            model.eval()
            all_preds, all_true = [], []
            with torch.no_grad():
                for batch in val_loader:
                    x = batch["feature"].float()
                    y = batch["label"].long()
                    logits = model(x)
                    preds = torch.argmax(logits, dim=1)
                    all_preds.extend(preds.cpu().numpy())
                    all_true.extend(y.cpu().numpy())

            plot_cnn_confusion_matrix(
                y_true=all_true,
                y_pred=all_preds,
                class_map=class_map,
                reports_dir="reports",
                tag=tag,
            )

            plot_cnn_f1_per_class(
                y_true=all_true,
                y_pred=all_preds,
                class_map=class_map,
                reports_dir="reports",
                tag=tag,
            )

    except Exception as e:
        print(f"[!] Training interrupted or failed: {e}")
        
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
                    probs = result.get("probabilities", {})

                    # Formatowanie słownika prawdopodobieństw do czytelnego stringa
                    # np. "spotify: 85.0%, youtube: 10.5%, inne: 4.5%"
                    probs_str = ", ".join([f"{label}: {prob * 100:.1f}%" for label, prob in probs.items()])

                    print(
                        f"[LIVE] Flow {key[4]}: {key[0]}:{key[2]} <-> {key[1]}:{key[3]} | "
                        f"App: {pred} ({conf * 100:.1f}%) | Szczegóły: [{probs_str}]"
                    )

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


def menu_online_mode_cnn():
    """Option 8: Live packet classification using the trained CNN model (Per-Packet)."""
    print("\n--- Online Classification (Real-time CNN) ---")

    # 1. Define required file paths (using your clean DATA_CNN_DIR layout)
    model_path = os.path.join(MODELS_DIR, "cnn_model_master.ckpt")
    class_map_path = os.path.join(DATA_CNN_DIR, "cnn_class_map.json")

    if not os.path.exists(model_path) or not os.path.exists(class_map_path):
        print(f"[!] Required files not found. Ensure cnn_model_master.ckpt and cnn_class_map.json exist.")
        return

    # 2. Initialize the live CNN classifier
    classifier = CNNOnline(model_path, class_map_path)
    if classifier.model is None:
        return

    sniffer = SnifferOnline()

    # 3. Define the async background worker
    def packet_processor():
        print("[*] CNN Worker thread started: processing packets live packet-by-packet...")
        while True:
            try:
                # Retrieve packet from the sniffer queue
                packet = sniffer.packet_queue.get(timeout=1.0)
                if packet is None:
                    break

                # Perform instant per-packet classification
                app_name, confidence = classifier.predict_packet(packet)

                if app_name is not None:
                    # Parse basic metadata from the packet layers for clear user display
                    from scapy.layers.inet import IP, TCP, UDP
                    from scapy.layers.inet6 import IPv6

                    proto = "TCP" if packet.haslayer(TCP) else "UDP"
                    sport = packet[TCP].sport if packet.haslayer(TCP) else packet[UDP].sport
                    dport = packet[TCP].dport if packet.haslayer(TCP) else packet[UDP].dport

                    if packet.haslayer(IP):
                        src, dst = packet[IP].src, packet[IP].dst
                    elif packet.haslayer(IPv6):
                        src, dst = packet[IPv6].src, packet[IPv6].dst
                    else:
                        src, dst = "unknown", "unknown"

                    # Output classification results immediately to the screen
                    print(
                        f"[LIVE-CNN] {src}:{sport} -> {dst}:{dport} ({proto}) | App: {app_name.upper()} ({confidence * 100:.1f}%)")

                sniffer.packet_queue.task_done()
            except queue.Empty:
                pass
            except Exception as e:
                print(f"[ERROR] {e}")

    # 4. Start the processing worker in a daemon thread
    processor_thread = threading.Thread(target=packet_processor, daemon=True)
    processor_thread.start()

    # 5. Start packet capturing on the sniffer side (blocks main thread until Ctrl+C)
    try:
        sniffer.start_capture()
    except KeyboardInterrupt:
        print("\n[*] Stopping live CNN capture...")
        sniffer.stop_capture()
        processor_thread.join(timeout=1.0)


def main():
    """Main application loop."""
    while True:
        print("\n" + "=" * 55)
        print(" NETWORK TRAFFIC CLASSIFIER - INTEGRATED SYSTEM ")
        print("=" * 55)
        print(" 1. Collect Training Data (Sniffer)")
        print(" 2. Extract Features (PCAP to CSV) [Random Forest]")
        print(" 3. Extract Features (PCAP to NPZ) [CNN]")
        print(" 4. Validate Datasets (Quality Report) [Random Forest]")
        print(" 5. Train Random Forest Models")
        print(" 6. Train CNN Models (PyTorch Lightning)")
        print(" 7. Run Online Classification (Live) [Random Forest]")
        print(" 8. Run Online Classification (Live) [CNN]")  # <--- DODANE
        print(" 0. Exit")

        cmd = input("\nSelect option: ").strip()
        if cmd == '1':
            menu_collect_data()
        elif cmd == '2':
            menu_extract_features()
        elif cmd == '3':
            menu_extract_features_cnn()
        elif cmd == '4':
            menu_validate_datasets()
        elif cmd == '5':
            menu_train_models()
        elif cmd == '6':
            menu_train_cnn_models()
        elif cmd == '7':
            menu_online_mode()
        elif cmd == '8':
            menu_online_mode_cnn()  # <--- PODPIĘCIE FUNKCJI
        elif cmd == '0':
            break


if __name__ == "__main__":
    for d in [DATA_RAW_DIR, DATA_CSV_DIR, MODELS_DIR, DATA_CNN_DIR, "reports"]:
        os.makedirs(d, exist_ok=True)
    main()