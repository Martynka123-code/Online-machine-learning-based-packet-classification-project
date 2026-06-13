import joblib
import pandas as pd
from collections import Counter
from scapy.utils import PcapReader
import time

from preprocessing.feature_extractor import FlowFeatureExtractor
from preprocessing.flow_keys import get_flow_key

def simulate_online_pcap(pcap_path, model_path, agg_mode="packet", agg_value=50, threshold=0.60):
    print(f"[*] Ładowanie modelu z {model_path}...")
    try:
        model_payload = joblib.load(model_path)
        model = model_payload["model"]
        feature_names = model_payload["feature_names"]
    except Exception as e:
        print(f"[!] Błąd ładowania modelu: {e}")
        return

    extractor = FlowFeatureExtractor(
        pcap_path=pcap_path,
        agg_mode=agg_mode, 
        agg_value=agg_value
    )

    active_flows = {}
    predictions_counter = Counter()

    print(f"[*] Rozpoczynam symulację LIVE na pliku: {pcap_path}")
    print(f"[*] Tryb: {agg_mode}, Wartość: {agg_value}")
    
    start_time = time.time()
    packet_count = 0

    with PcapReader(pcap_path) as packets:
        for packet in packets:
            packet_count += 1
            
            # 1. Pobranie klucza 
            flow_key = get_flow_key(packet)
            if not flow_key:
                continue

            if flow_key not in active_flows:
                active_flows[flow_key] = []
            
            active_flows[flow_key].append(packet)
            flow_packets = active_flows[flow_key]

            # 2. Sprawdzenie warunku flusha
            flush_flow = False
            if agg_mode == "packet":
                flush_flow = len(flow_packets) >= agg_value
            elif agg_mode == "time":
                # Symulacja czasu pcapa (czas wirtualny)
                time_diff = float(flow_packets[-1].time - flow_packets[0].time)
                flush_flow = time_diff >= agg_value

            # 3. Predykcja
            if flush_flow:
                features = extractor.calculate_features(flow_packets)
                df_eval = pd.DataFrame([features])
                
                for col in feature_names:
                    if col not in df_eval.columns:
                        df_eval[col] = 0.0
                
                df_eval = df_eval[feature_names] 
                
                probs = model.predict_proba(df_eval)[0]
                max_prob = max(probs)
                
                if max_prob < threshold:
                    predicted_class = "OTHER"
                else:
                    predicted_class = model.classes_[probs.argmax()]
                
                predictions_counter[predicted_class] += 1
                print(f"[+] Flow: {flow_key[3]}<->{flow_key[4]} | Pred: {predicted_class} ({max_prob*100:.1f}%)")
                
                active_flows[flow_key] = []

    elapsed = time.time() - start_time
    print("\n" + "="*40)
    print(" PODSUMOWANIE SYMULACJI ")
    print("="*40)
    print(f"Przeanalizowano pakietów: {packet_count}")
    print(f"Czas symulacji: {elapsed:.2f} s")
    print("Rozkład predykcji:")
    for app, count in predictions_counter.most_common():
        print(f" - {app}: {count} agregacji")
    print("="*40)

if __name__ == "__main__":
    simulate_online_pcap(
        pcap_path="data/additional_raw/discord_camera_2026-05-30.pcap", 
        model_path="models/saved_models/rf_model.pkl", 
        agg_mode="packet", 
        agg_value=50,
        threshold=0.65
    )