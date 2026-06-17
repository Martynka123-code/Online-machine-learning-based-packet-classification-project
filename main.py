from datetime import datetime
import os
import time
import threading
import torch
import queue
import json
from scapy.all import sniff
import numpy as np

from pytorch_lightning.loggers import CSVLogger
from sklearn.metrics import classification_report

from visualization.live_dashboard import LiveAccuracyDashboard

from pytorch_lightning.callbacks import LearningRateMonitor
from visualization.confusion_matrix_plot import plot_confusion_matrix
from models.cnn_trainer import OptimizedPacketCNN, PacketByteDataset, MetricsCallback
from visualization.cnn_visualizer import (
    plot_cnn_training_curves,
    plot_cnn_class_distribution,
    plot_cnn_confusion_matrix,
    plot_cnn_f1_per_class,
)
# POPRAWKA: usunięto zduplikowany import Trainer i CSVLogger (były zaimportowane dwa razy)
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from config import DATA_CNN_DIR, DATA_RAW_DIR, DATA_CSV_DIR, MODELS_DIR, GRANULARITIES, TIME_WINDOWS
from models.cnn_online import CNNOnline
from preprocessing.cnn_preprocessor import CNNPreprocessor
from sniffing.sniffer_training import SnifferTraining
from sniffing.sniffer_online import SnifferOnline
from preprocessing.feature_extractor import FlowFeatureExtractor
from models.rf_trainer import RandomForestTrainer
from models.rf_online import RandomForestOnline
from torch.utils.data import DataLoader, random_split
from visualization.rf_visualizer import plot_granularity_comparison


def menu_collect_data():
    """Option 1: Collect raw traffic using the process-to-port sniffer."""
    print("\n--- Data Collection Mode ---")
    app_name = input("Enter application name to monitor (e.g., spotify): ").strip().lower()
    if not app_name:
        return

    sniffer = SnifferTraining(target_apps=[app_name])
    sniffer.start()


def menu_extract_features():
    """Option 2: Convert PCAP files to CSV datasets for multiple granularities."""
    print("\n--- Feature Extraction (PCAP to CSV) ---")

    balanced_dir = os.path.join("data", "balanced_pcaps")
    if not os.path.exists(balanced_dir):
        print(f"[!] Folder {balanced_dir} nie istnieje. Najpierw wygeneruj zbalansowane PCAPy (pcap_balancer.py).")
        return

    files = sorted([f for f in os.listdir(balanced_dir) if f.endswith('.pcap')])
    if not files:
        print(f"[!] Brak plików PCAP w {balanced_dir}")
        return

    print("\nDostępne pliki PCAP w zbalansowanym folderze:")
    for f in files:
        print(f"  - {f}")

    print("\n[WSKAZÓWKA] Wpisz 'all', aby automatycznie przetworzyć WSZYSTKIE pliki.")
    user_input = input("Wprowadź nazwę pliku (lub 'all'): ").strip()

    if user_input.lower() == 'all':
        print("\n[*] Rozpoczynam automatyczne przetwarzanie wszystkich plików (Tryb ALL)...")

        confirm = input("Czy wyczyścić stare pliki CSV przed ekstrakcją? (y/n - zalecane 'y'): ").strip().lower()
        if confirm == 'y':
            for f in os.listdir(DATA_CSV_DIR):
                if f.endswith(".csv"):
                    os.remove(os.path.join(DATA_CSV_DIR, f))
            print("[+] Stare pliki CSV zostały usunięte. Ekstrakcja na czysto.")

        for f in files:
            pcap_path = os.path.join(balanced_dir, f)
            label = f.split('_')[0].capitalize()
            print(f"\n---> Przetwarzanie pliku: {f} | Wykryta Etykieta: {label}")

            print(f"  [*] Ekstrakcja dla granularności (pakiety): {GRANULARITIES}")
            for gran in GRANULARITIES:
                output_csv = os.path.join(DATA_CSV_DIR, f"rf_dataset_{gran}.csv")
                extractor = FlowFeatureExtractor(pcap_path=pcap_path, label=label, agg_mode='packet', agg_value=gran)
                extractor.process_and_save(output_csv)

            print(f"  [*] Ekstrakcja dla okien czasowych (sekundy): {TIME_WINDOWS}")
            for tw in TIME_WINDOWS:
                output_csv_time = os.path.join(DATA_CSV_DIR, f"rf_dataset_time_{tw}.csv")
                extractor = FlowFeatureExtractor(pcap_path=pcap_path, label=label, agg_mode='time', agg_value=tw)
                extractor.process_and_save(output_csv_time)

        print("\n[+] ZAKOŃCZONO MASOWĄ EKSTRAKCJĄ CECH!")
        return

    else:
        pcap_path = os.path.join(balanced_dir, user_input)
        if not os.path.exists(pcap_path):
            print("[!] Nie znaleziono pliku.")
            return

        guessed_label = user_input.split('_')[0].capitalize()
        label = input(f"Wprowadź etykietę dla ruchu (wciśnij Enter by użyć '{guessed_label}'): ").strip()
        if not label:
            label = guessed_label

        print(f"[*] Ekstrakcja cech dla granularności: {GRANULARITIES}")
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

    detected_apps = sorted(list(set([os.path.splitext(f)[0].split('_')[0] for f in files])))
    class_map = {app_name: idx for idx, app_name in enumerate(detected_apps)}

    print("\n[+] Automatically discovered applications and assigned indices:")
    for app_name, idx in class_map.items():
        print(f"  Class {idx}: {app_name}")

    map_path = os.path.join(DATA_CNN_DIR, "cnn_class_map.json")
    with open(map_path, 'w') as f:
        json.dump(class_map, f, indent=4)
    print(f"[+] Class mapping dictionary saved to: {map_path}")

    # Inicjalizujemy preprocesor RAZ, bo chcemy mieć jeden wspólny zbiór (self.data)
    preprocessor = CNNPreprocessor(max_length=1000)
    print("\n[*] Processing PCAP files into a combined memory array...")

    for f in files:
        app_name = os.path.splitext(f)[0].split('_')[0]
        label_idx = class_map[app_name]
        pcap_path = os.path.join(DATA_RAW_DIR, f)
        
        # 1. Wykrywamy lokalne IP konkretnie dla TEGO pliku PCAP
        local_ips = FlowFeatureExtractor(pcap_path=pcap_path)._detect_local_ips_from_pcap(pcap_path)

        # 2. AKTUALIZACJA: Podajemy preprocesorowi nowe IP przed procesowaniem!
        # Dzięki temu kodowanie kierunku w CNNPreprocessor wie, kto jest "klientem" w tym konkretnym PCAPie
        preprocessor.local_ips = local_ips

        # 3. Ekstrakcja z użyciem w/w adresów IP
        preprocessor.process_pcap(pcap_path, label_idx)

    output_npz = os.path.join(DATA_CNN_DIR, f"cnn_dataset_{datetime.now().strftime('%Y-%m-%d')}_master.npz")
    preprocessor.save_dataset(output_npz)

def menu_validate_datasets():
    """Option 4: Quality check for generated CSV datasets."""
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
    """Option 5: Train RF models for both Packet and Time granularities."""
    print("\n--- Model Training & Advanced Analytics ---")
    results_pkt = {}
    results_time = {}

    for gran in GRANULARITIES:
        csv_path = os.path.join(DATA_CSV_DIR, f"rf_dataset_{gran}.csv")
        if os.path.exists(csv_path):
            print(f"\n>>> Training for Granularity: {gran} packets")
            trainer = RandomForestTrainer(csv_path, reports_dir="reports")
            acc = trainer.train_and_evaluate()
            if acc is not None:
                results_pkt[gran] = acc
                model_path = os.path.join(MODELS_DIR, f"rf_model_{gran}.pkl")
                trainer.save_model(model_path)

    for tw in TIME_WINDOWS:
        csv_path = os.path.join(DATA_CSV_DIR, f"rf_dataset_time_{tw}.csv")
        if os.path.exists(csv_path):
            print(f"\n>>> Training for Time Window: {tw} seconds")
            trainer = RandomForestTrainer(csv_path, reports_dir="reports")
            acc = trainer.train_and_evaluate()
            if acc is not None:
                results_time[tw] = acc
                model_path = os.path.join(MODELS_DIR, f"rf_model_time_{tw}.pkl")
                trainer.save_model(model_path)

    if results_pkt:
        print("\n--- Training Summary (Packets) ---")
        for key, a in results_pkt.items():
            print(f"Config {key:>3} pkts: Accuracy {a:.2%}")
        try:
            plot_granularity_comparison(results_pkt)
        except Exception as e:
            print(f"[!] Nie udało się narysować wykresu pakietów: {e}")

    if results_time:
        print("\n--- Training Summary (Time Windows) ---")
        for key, a in results_time.items():
            print(f"Config {key:>3} s: Accuracy {a:.2%}")


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

    class_map_path = os.path.join(DATA_CNN_DIR, "cnn_class_map.json")
    if not os.path.exists(class_map_path):
        print(f"[!] Class map not found at {class_map_path}. Run option 3 first.")
        return

    with open(class_map_path, "r") as f:
        class_map = json.load(f)

    try:
        full_dataset = PacketByteDataset(dataset_path)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
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

    model_base_name = os.path.splitext(dataset_file)[0].replace("cnn_dataset_", "")

    metrics_cb = MetricsCallback()
    csv_logger = CSVLogger(save_dir="lightning_logs", name=f"cnn_{model_base_name}")

    print("\n[*] Starting PyTorch Lightning Training Session...")
    trainer = Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices=1,
        logger=csv_logger,
        callbacks=[metrics_cb, LearningRateMonitor(logging_interval='epoch')],
    )

    try:
        trainer.fit(model, train_loader, val_loader)
    except Exception as e:
        print(f"[!] Training interrupted or failed: {e}")
        return

    model_save_path = os.path.join(MODELS_DIR, f"cnn_model_{model_base_name}.ckpt")
    trainer.save_checkpoint(model_save_path)
    print(f"\n[+] CNN Training complete! Checkpoint saved to: {model_save_path}")

    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)

    history = metrics_cb.history
    if history["train_loss"] and history["val_loss"] and history["val_acc"]:
        plot_cnn_training_curves(
            train_losses=history["train_loss"],
            val_losses=history["val_loss"],
            val_accuracies=history["val_acc"],
            reports_dir=reports_dir,
            tag=model_base_name,
            val_f1_scores=history["val_f1_macro"] or None,
        )
    else:
        print("[!] No epoch metrics collected — skipping training curves plot.")

    plot_cnn_class_distribution(
        labels=full_dataset.labels,
        class_map=class_map,
        reports_dir=reports_dir,
        tag=model_base_name,
    )

    print("\n[*] Running inference on validation split for confusion matrix / F1 plots...")
    model.eval()
    torch.set_grad_enabled(False)

    y_true, y_pred = [], []
    for batch in val_loader:
        x = batch["feature"]
        y = batch["label"].long()
        logits = model(x)
        preds = torch.argmax(logits, dim=1)
        y_true.extend(y.tolist())
        y_pred.extend(preds.tolist())

    torch.set_grad_enabled(True)

    plot_cnn_confusion_matrix(y_true, y_pred, class_map, reports_dir, tag=model_base_name)
    plot_cnn_f1_per_class(y_true, y_pred, class_map, reports_dir, tag=model_base_name)

    print(f"\n[+] All training reports saved in: {reports_dir}/")


def menu_online_mode():
    """Option 7: Live classification with asynchronous packet processing buffer."""
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
        print(f"[!] Model '{model_name}' not found. Train it first (Option 5).")
        return

    classifier = RandomForestOnline(model_path)

    dashboard = LiveAccuracyDashboard(title="Live Monitor - Random Forest")

    def on_flow_ready(features, flow_key):
        result = classifier.classify_stream(features)
        pred = result.get("prediction", "UNKNOWN")
        conf = result.get("confidence", 0.0)
        probs = result.get("probabilities", {})
        probs_str = ", ".join([f"{label}: {prob * 100:.1f}%" for label, prob in probs.items()])
        print(
            f"[LIVE] Flow {flow_key[4]}: {flow_key[0]}:{flow_key[2]} <-> {flow_key[1]}:{flow_key[3]} | "
            f"App: {pred} ({conf * 100:.1f}%) | Szczegóły: [{probs_str}]"
        )
        dashboard.push(pred, conf, meta=f"{flow_key[0]}:{flow_key[2]} <-> {flow_key[1]}:{flow_key[3]}")

    print(f"[*] Loaded model: {model_path}")
    sniffer = SnifferOnline(agg_mode=agg_mode, agg_value=agg_value,
                            mode="flow", prediction_callback=on_flow_ready)

    from scapy.all import get_if_list, get_if_addr
    print("Dostępne interfejsy:")
    for iface in get_if_list():
        try:
            print(f"  {iface}: {get_if_addr(iface)}")
        except Exception:
            pass

    capture_thread = threading.Thread(target=sniffer.start_capture, daemon=True)
    capture_thread.start()

    try:
        dashboard.run()
    finally:
        sniffer.stop_capture()


def menu_online_mode_cnn():
    """Option 8: Live packet classification using the trained CNN model (Per-Packet)."""
    print("\n--- Online Classification (Real-time CNN) ---")

    model_path = os.path.join(MODELS_DIR, "cnn_model_master.ckpt")
    class_map_path = os.path.join(DATA_CNN_DIR, "cnn_class_map.json")

    if not os.path.exists(model_path) or not os.path.exists(class_map_path):
        print(f"[!] Required files not found. Ensure cnn_model_master.ckpt and cnn_class_map.json exist.")
        return

    classifier = CNNOnline(model_path, class_map_path)
    if classifier.model is None:
        return

    from visualization.live_dashboard import LiveAccuracyDashboard
    dashboard = LiveAccuracyDashboard(title="Live Monitor - CNN")

    from scapy.layers.inet import IP, TCP, UDP
    from scapy.layers.inet6 import IPv6

    def on_packet(packet):
        app_name, confidence = classifier.predict_packet(packet)
        if app_name is None:
            return

        proto = "TCP" if packet.haslayer(TCP) else "UDP"
        sport = packet[TCP].sport if packet.haslayer(TCP) else packet[UDP].sport
        dport = packet[TCP].dport if packet.haslayer(TCP) else packet[UDP].dport

        if packet.haslayer(IP):
            src, dst = packet[IP].src, packet[IP].dst
        elif packet.haslayer(IPv6):
            src, dst = packet[IPv6].src, packet[IPv6].dst
        else:
            src, dst = "unknown", "unknown"

        print(f"[LIVE-CNN] {src}:{sport} -> {dst}:{dport} ({proto}) | App: {app_name.upper()} ({confidence * 100:.1f}%)")
        dashboard.push(app_name, confidence, meta=f"{src}:{sport} -> {dst}:{dport} ({proto})")

    sniffer = SnifferOnline(mode="raw", packet_callback=on_packet)

    capture_thread = threading.Thread(target=sniffer.start_capture, daemon=True)
    capture_thread.start()

    try:
        dashboard.run()
    finally:
        sniffer.stop_capture()


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
        print(" 8. Run Online Classification (Live) [CNN]")
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
            menu_online_mode_cnn()
        elif cmd == '0':
            break


if __name__ == "__main__":
    for d in [DATA_RAW_DIR, DATA_CSV_DIR, MODELS_DIR, DATA_CNN_DIR, "reports"]:
        os.makedirs(d, exist_ok=True)
    main()