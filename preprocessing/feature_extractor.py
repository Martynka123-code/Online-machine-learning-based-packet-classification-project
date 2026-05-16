import os
import numpy as np
import pandas as pd
from scapy.all import rdpcap
from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.inet6 import IPv6
import warnings

warnings.filterwarnings('ignore')  

class FlowFeatureExtractor:
    def __init__(self, pcap_path=None, label=None, granularity=100, time_window=1.0):
        """
        pcap_path: Path to the .pcap file (None for Online mode)
        label: Traffic class label (None for Online mode)
        time_window: Max time in seconds to wait before flushing the flow aggregate
        """
        self.pcap_path = pcap_path
        self.label = label
        self.granularity = granularity
        self.time_window = time_window 
        self.flows = {}
        self.dataset = []

    def _get_flow_key(self, packet):
        """Returns a direction-agnostic 5-tuple flow key. Supports IPv4 & IPv6."""
        if packet.haslayer(IP):
            ip_layer = packet[IP]
            proto = ip_layer.proto
        elif packet.haslayer(IPv6):
            ip_layer = packet[IPv6]
            proto = ip_layer.nh
        else:
            return None

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
            "granularity": len(packets)
        }
        
        if self.label is not None:
            features["label"] = self.label
            
        return features

    def process_and_save(self, output_csv, batch_size=10000):
        if self.pcap_path is None:
            print("[!] Error: No PCAP path provided. This method is for offline extraction.")
            return

        print(f"[*] Extracting features from: {self.pcap_path} | Granularity: {self.granularity} | Time Window: {self.time_window}s")
        
        from scapy.utils import PcapReader
        with PcapReader(self.pcap_path) as packets:
            for pkt in packets:
                flow_key = self._get_flow_key(pkt)
                if not flow_key: continue

                if flow_key not in self.flows: 
                    self.flows[flow_key] = []
                    
                self.flows[flow_key].append(pkt)

                flow_packets = self.flows[flow_key]
                time_diff = float(flow_packets[-1].time - flow_packets[0].time)

                if len(flow_packets) >= self.granularity or time_diff >= self.time_window:
                    self.dataset.append(self._calculate_features(flow_packets))
                    self.flows[flow_key] = []  

                    if len(self.dataset) >= batch_size:
                        self._save_batch(output_csv)

        if self.dataset:
            self._save_batch(output_csv)
        else:
            print("[!] Not enough packets to create a single aggregate.")

    def _save_batch(self, output_csv):
        df = pd.DataFrame(self.dataset).round(5)
        file_exists = os.path.isfile(output_csv)
        df.to_csv(output_csv, mode='a', header=not file_exists, index=False)
        print(f"[+] Saved batch of {len(self.dataset)} aggregates to {output_csv}")
        self.dataset.clear()