# preprocessing/feature_extractor.py
import os
import numpy as np
import pandas as pd
from scapy.all import rdpcap
from scapy.layers.inet import IP, TCP, UDP
import warnings

warnings.filterwarnings('ignore')  # Ukrywa ostrzeżenia Scapy


class FlowFeatureExtractor:
    def __init__(self, pcap_path, label, granularity=100):
        self.pcap_path = pcap_path
        self.label = label
        self.granularity = granularity
        self.flows = {}
        self.dataset = []

    def _get_flow_key(self, packet):
        """Zwraca zanonimizowany klucz przepływu (5-tuple), ignorując kierunek."""
        if not packet.haslayer(IP): return None
        ip_layer = packet[IP]
        proto = ip_layer.proto
        if packet.haslayer(TCP):
            sport, dport = packet[TCP].sport, packet[TCP].dport
        elif packet.haslayer(UDP):
            sport, dport = packet[UDP].sport, packet[UDP].dport
        else:
            return None

        ips = tuple(sorted([ip_layer.src, ip_layer.dst]))
        ports = tuple(sorted([sport, dport]))
        return (ips[0], ips[1], ports[0], ports[1], proto)

    def _get_payload_length(self, packet):
        if packet.haslayer(TCP) and packet[TCP].payload:
            return len(packet[TCP].payload)
        elif packet.haslayer(UDP) and packet[UDP].payload:
            return len(packet[UDP].payload)
        return 0

    def _calculate_features(self, packets):
        """Wylicza zdefiniowane cechy z agregatu, MASKUJĄC adresy IP/MAC."""
        pkt_lengths = [len(p) for p in packets]
        payload_lengths = [self._get_payload_length(p) for p in packets]
        iats = [float(packets[i].time - packets[i - 1].time) for i in range(1, len(packets))]
        if not iats: iats = [0.0]

        iat_mean = np.mean(iats)
        iat_std = np.std(iats)
        iat_cov = (iat_std / iat_mean) if iat_mean > 0 else 0.0

        features = {
            "pkt_len_max": np.max(pkt_lengths),
            "pkt_len_std": np.std(pkt_lengths),
            "pkt_len_p50": np.percentile(pkt_lengths, 50),
            "pkt_len_p75": np.percentile(pkt_lengths, 75),
            "payload_median": np.median(payload_lengths),
            "iat_std": iat_std,
            "iat_p75_minus_p50": np.percentile(iats, 75) - np.median(iats),
            "iat_p95_minus_p50": np.percentile(iats, 95) - np.median(iats),
            "iat_cov_std_mean": iat_cov,
            "iat_median": np.median(iats),
            "granularity": self.granularity,
            "label": self.label
        }
        return features

    def process_and_save(self, output_csv):
        print(f"[*] Ekstrakcja cech z: {self.pcap_path} | Granularność: {self.granularity}")
        packets = rdpcap(self.pcap_path)

        for pkt in packets:
            flow_key = self._get_flow_key(pkt)
            if not flow_key: continue

            if flow_key not in self.flows: self.flows[flow_key] = []
            self.flows[flow_key].append(pkt)

            if len(self.flows[flow_key]) == self.granularity:
                self.dataset.append(self._calculate_features(self.flows[flow_key]))
                self.flows[flow_key] = []  # Reset agregatu

        if not self.dataset:
            print("[!] Brak wystarczającej liczby pakietów do stworzenia agregatu.")
            return

        df = pd.DataFrame(self.dataset).round(5)
        file_exists = os.path.isfile(output_csv)
        df.to_csv(output_csv, mode='a', header=not file_exists, index=False)
        print(f"[+] Zapisano {len(self.dataset)} agregatów do {output_csv}")