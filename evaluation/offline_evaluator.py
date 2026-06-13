"""
offline_evaluator.py
--------------------
Główna logika ewaluacji modeli RF i CNN na plikach PCAP.
Zwraca słownik wynikowy - raportowanie jest w report_generator.py
"""

import os
import numpy as np
from collections import Counter
from scapy.all import rdpcap

from config import DATA_RAW_DIR, MODELS_DIR, DATA_CNN_DIR
from preprocessing.feature_extractor import FlowFeatureExtractor
from models.rf_online import RandomForestOnline
from models.cnn_online import CNNOnline


def evaluate_rf(packets, model_path, agg_mode, agg_value):
    """
    Klasyfikuje pakiety modelem Random Forest.

    Returns:
        list of (prediction, confidence)
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Nie znaleziono modelu RF: {model_path}")

    model = RandomForestOnline(model_path)
    extractor = FlowFeatureExtractor(
    pcap_path=pcap_path, label=None,
    agg_mode=agg_mode, agg_value=agg_value
    )

    results = []
    flows = {}

    for pkt in packets:
        key = extractor._get_flow_key(pkt)
        if not key:
            continue

        flows.setdefault(key, []).append(pkt)
        flow_packets = flows[key]
        flush_flow = False

        if agg_mode == "packet":
            flush_flow = len(flow_packets) >= agg_value
        elif agg_mode == "time":
            time_diff = float(flow_packets[-1].time - flow_packets[0].time)
            flush_flow = time_diff >= agg_value

        if flush_flow:
            features = extractor._calculate_features(flow_packets)
            result = model.classify_stream(features)
            results.append((
                result.get("prediction", "UNKNOWN").lower(),
                result.get("confidence", 0.0)
            ))
            flows[key] = []

    return results


def evaluate_cnn(packets, model_path, class_map_path):
    """
    Klasyfikuje pakiety modelem CNN (per-pakiet).

    Returns:
        list of (prediction, confidence)
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Nie znaleziono modelu CNN: {model_path}")
    if not os.path.exists(class_map_path):
        raise FileNotFoundError(f"Nie znaleziono mapy klas CNN: {class_map_path}")

    model = CNNOnline(model_path, class_map_path)
    if model.model is None:
        raise RuntimeError("Nie udało się załadować modelu CNN.")

    results = []
    for pkt in packets:
        app_name, conf = model.predict_packet(pkt)
        if app_name is not None:
            results.append((app_name.lower(), conf))

    return results


def evaluate_pcap(pcap_path, true_label, model_type, model_params):
    """
    Ewaluuje jeden plik PCAP i zwraca słownik ze statystykami.

    Args:
        pcap_path (str):     Ścieżka do pliku .pcap
        true_label (str):    Prawdziwa klasa ruchu (np. "spotify")
        model_type (str):    "rf" lub "cnn"
        model_params (dict): Parametry modelu:
                             - RF: {"agg_mode": "packet"/"time", "agg_value": int/float}
                             - CNN: (brak dodatkowych - ścieżki z configu)

    Returns:
        dict ze statystykami lub None przy błędzie
    """
    true_label = true_label.strip().lower()
    pcap_name = os.path.basename(pcap_path)

    print(f"  [*] Wczytywanie: {pcap_name} ...")
    packets = rdpcap(pcap_path)
    print(f"  [*] Załadowano {len(packets)} pakietów.")

    raw_results = []

    if model_type == "rf":
        agg_mode = model_params["agg_mode"]
        agg_value = model_params["agg_value"]
        model_name = (
            f"rf_model_{agg_value}.pkl"
            if agg_mode == "packet"
            else f"rf_model_time_{agg_value}.pkl"
        )
        model_path = os.path.join(MODELS_DIR, model_name)
        print(f"  [*] Klasyfikacja RF [{agg_mode}, val={agg_value}] ...")
        raw_results = evaluate_rf(packets, model_path, agg_mode, agg_value)

    elif model_type == "cnn":
        model_path = os.path.join(MODELS_DIR, "cnn_model_master.ckpt")
        class_map_path = os.path.join(DATA_CNN_DIR, "cnn_class_map.json")
        print(f"  [*] Klasyfikacja CNN (per-pakiet) ...")
        raw_results = evaluate_cnn(packets, model_path, class_map_path)

    if not raw_results:
        print(f"  [!] Brak wyników dla {pcap_name}.")
        return None

    predictions = [r[0] for r in raw_results]
    confidences = [r[1] for r in raw_results]
    counts = Counter(predictions)
    total = len(predictions)

    correct_count = counts.get(true_label, 0)
    other_count = sum(counts.get(k, 0) for k in ("other", "unknown"))
    wrong_count = total - correct_count - other_count

    # Pewność per-klasa (tylko RF przekazuje sensowne confidence)
    class_confidence = {}
    if model_type == "rf":
        conf_buckets = {}
        for pred, conf in zip(predictions, confidences):
            conf_buckets.setdefault(pred, []).append(conf)
        class_confidence = {cls: float(np.mean(vals)) for cls, vals in conf_buckets.items()}

    return {
        "pcap_name": pcap_name,
        "true_label": true_label,
        "model_type": model_type,
        "model_params": model_params,
        "total": total,
        "correct": correct_count,
        "wrong": wrong_count,
        "other": other_count,
        "counts": dict(counts),
        "mean_confidence": float(np.mean(confidences)) if confidences else 0.0,
        "class_confidence": class_confidence,
    }