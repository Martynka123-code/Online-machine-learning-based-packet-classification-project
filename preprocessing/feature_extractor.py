import os
import warnings
import psutil
import socket
from collections import Counter

warnings.filterwarnings("ignore")

from preprocessing.direction import split_by_direction
from preprocessing.statistics import direction_features
from preprocessing.tcp_features import tcp_flag_counts, retransmission_count
from preprocessing.burst import burst_features
from preprocessing.offline_processor import OfflineProcessor
from preprocessing.flow_keys import get_flow_key
from scapy.layers.inet import IP
from scapy.layers.inet6 import IPv6
from scapy.utils import PcapReader


class FlowFeatureExtractor:

    def __init__(
            self,
            pcap_path=None,
            label=None,
            agg_mode="packet",
            agg_value=100
    ):
        self.pcap_path = pcap_path
        self.label = label
        self.agg_mode = agg_mode
        self.agg_value = agg_value

        if self.pcap_path is not None:
            self.local_ips = self._detect_local_ips_from_pcap(self.pcap_path)
        else:
            self.local_ips = self._get_local_ips()

    def _get_local_ips(self) -> set:
        ips = set()
        for addrs in psutil.net_if_addrs().values():
            for addr in addrs:
                if addr.family in (socket.AF_INET, socket.AF_INET6):
                    if not addr.address.startswith("127.") and addr.address != "::1":
                        ips.add(addr.address.split('%')[0])
        return ips

    def _detect_local_ips_from_pcap(self, pcap_path) -> set:
        ipv4_counts = Counter()
        ipv6_counts = Counter()

        with PcapReader(pcap_path) as reader:
            for pkt in reader:
                if pkt.haslayer(IP):
                    ipv4_counts[pkt[IP].src] += 1
                    ipv4_counts[pkt[IP].dst] += 1
                elif pkt.haslayer(IPv6):
                    ipv6_counts[pkt[IPv6].src] += 1
                    ipv6_counts[pkt[IPv6].dst] += 1

        local_ips = set()
        if ipv4_counts:
            local_ips.add(ipv4_counts.most_common(1)[0][0])
        if ipv6_counts:
            local_ips.add(ipv6_counts.most_common(1)[0][0])

        if local_ips:
            print(f"[+] [{pcap_path}] Wykryty lokalny IP: {local_ips}")
        else:
            print(f"[!] [{pcap_path}] Nie wykryto lokalnego IP — fwd/bwd ratio = 0.5/0.5.")

        return local_ips

    def _get_flow_key(self, packet):
        return get_flow_key(packet)

    def calculate_features(self, packets):
        fwd_packets, bwd_packets = [], []

        for p in packets:
            if p.haslayer(IP):
                src_ip = p[IP].src
            elif p.haslayer(IPv6):
                src_ip = p[IPv6].src
            else:
                continue

            if src_ip in self.local_ips:
                fwd_packets.append(p)
            else:
                bwd_packets.append(p)

        features = {}
        features.update(direction_features(fwd_packets, "fwd"))
        features.update(direction_features(bwd_packets, "bwd"))

        fwd_bytes = sum(len(p) for p in fwd_packets)
        bwd_bytes = sum(len(p) for p in bwd_packets)
        total_bytes = fwd_bytes + bwd_bytes

        features["bytes_ratio_fwd"] = fwd_bytes / total_bytes if total_bytes > 0 else 0.5
        features["pkt_ratio_fwd"] = len(fwd_packets) / len(packets) if packets else 0.5

        flag_counts = tcp_flag_counts(packets)
        packet_count = max(len(packets), 1)

        for flag, count in flag_counts.items():
            features[f"tcp_{flag.lower()}_ratio"] = count / packet_count

        features["tcp_retrans_ratio"] = retransmission_count(packets) / packet_count
        features.update(burst_features(packets))

        udp_count = sum(1 for p in packets if p.haslayer("UDP"))
        features["udp_ratio"] = udp_count / packet_count

        features["actual_packets_in_flow"] = len(packets)
        features["agg_mode"] = self.agg_mode
        features["agg_value"] = self.agg_value

        if packets:
            features["flow_id"] = str(get_flow_key(packets[0]))
        else:
            features["flow_id"] = "unknown"

        # POPRAWKA: session_id = nazwa pliku pcap.
        # Umożliwia grupowanie per-sesja w GroupShuffleSplit zamiast per-flow.
        if self.pcap_path is not None:
            features["session_id"] = os.path.basename(self.pcap_path)

        if self.label is not None:
            features["label"] = self.label

        return features

    def _calculate_features(self, packets):
        """Wrapper dla kompatybilności wstecznej."""
        return self.calculate_features(packets)

    def process_and_save(self, output_csv, batch_size=10000):
        if self.pcap_path is None:
            raise ValueError("pcap_path cannot be None during offline extraction.")

        processor = OfflineProcessor(extractor=self)
        processor.process_and_save(output_csv=output_csv, batch_size=batch_size)