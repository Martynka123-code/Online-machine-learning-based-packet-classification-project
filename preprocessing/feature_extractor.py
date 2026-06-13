import warnings
import psutil
import socket

warnings.filterwarnings("ignore")

from preprocessing.direction import split_by_direction
from preprocessing.statistics import direction_features
from preprocessing.tcp_features import (
    tcp_flag_counts,
    retransmission_count,
)
from preprocessing.burst import burst_features
from preprocessing.offline_processor import OfflineProcessor
from preprocessing.flow_keys import get_flow_key
from scapy.layers.inet import IP
from scapy.layers.inet6 import IPv6


class FlowFeatureExtractor:
    """
    Main feature extraction class.

    Supports:
    - offline PCAP -> CSV extraction
    - online live packet classification
    - packet aggregation
    - time aggregation
    """

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
        
        # Automatyczne pobranie adresów IP tego komputera
        self.local_ips = self._get_local_ips()

    def _get_local_ips(self) -> set:
        """Pobiera wszystkie lokalne adresy IP (IPv4 i IPv6) tego komputera."""
        ips = set()
        for addrs in psutil.net_if_addrs().values():
            for addr in addrs:
                if addr.family in (socket.AF_INET, socket.AF_INET6):
                    # Pomijamy localhost (127.0.0.1)
                    if not addr.address.startswith("127.") and addr.address != "::1":
                        ips.add(addr.address.split('%')[0]) 
        return ips

    # ------------------------------------------------------------------
    # Compatibility for ONLINE mode (Sniffer)
    # ------------------------------------------------------------------

    def _get_flow_key(self, packet):
        """
        Compatibility method for online sniffer.
        Delegates to the standalone get_flow_key function.
        """
        return get_flow_key(packet)

    # ------------------------------------------------------------------
    # Main feature engineering
    # ------------------------------------------------------------------

    def calculate_features(self, packets):
        
        fwd_packets, bwd_packets = [], []
        
        for p in packets:
            if p.haslayer(IP):
                src_ip = p[IP].src
            elif p.haslayer(IPv6):
                src_ip = p[IPv6].src
            else:
                continue
                
            # Jeśli pakiet idzie od nas (nasze IP), to jest to FWD (Upload)
            if src_ip in self.local_ips:
                fwd_packets.append(p)
            else:
                bwd_packets.append(p)
                
        features = {}

        # --------------------------------------------------------------
        # Directional statistics
        # --------------------------------------------------------------

        features.update(
            direction_features(
                fwd_packets,
                "fwd"
            )
        )

        features.update(
            direction_features(
                bwd_packets,
                "bwd"
            )
        )

        # --------------------------------------------------------------
        # Upload / download ratios
        # --------------------------------------------------------------

        fwd_bytes = sum(
            len(packet)
            for packet in fwd_packets
        )

        bwd_bytes = sum(
            len(packet)
            for packet in bwd_packets
        )

        total_bytes = fwd_bytes + bwd_bytes

        features["bytes_ratio_fwd"] = (
            fwd_bytes / total_bytes
            if total_bytes > 0 else 0.5
        )

        features["pkt_ratio_fwd"] = (
            len(fwd_packets) / len(packets)
            if packets else 0.5
        )

        # --------------------------------------------------------------
        # TCP flag statistics
        # --------------------------------------------------------------

        flag_counts = tcp_flag_counts(
            packets
        )

        packet_count = max(
            len(packets),
            1
        )

        for flag, count in flag_counts.items():
            features[
                f"tcp_{flag.lower()}_ratio"
            ] = count / packet_count

        # --------------------------------------------------------------
        # TCP retransmissions
        # --------------------------------------------------------------

        features["tcp_retrans_ratio"] = (
                retransmission_count(packets)
                / packet_count
        )

        # --------------------------------------------------------------
        # Burst detection
        # --------------------------------------------------------------

        features.update(
            burst_features(packets)
        )
        # --------------------------------------------------------------
        # Transport Protocol Ratios (UDP)
        # --------------------------------------------------------------
        udp_count = 0
        for packet in packets:
            # Zakładając, że 'packets' to lista obiektów pakietów z biblioteki Scapy
            if packet.haslayer("UDP"):
                udp_count += 1

        features["udp_ratio"] = udp_count / packet_count

        # --------------------------------------------------------------
        # Metadata
        # --------------------------------------------------------------

        features[
            "actual_packets_in_flow"
        ] = len(packets)

        features["agg_mode"] = (
            self.agg_mode
        )

        features["agg_value"] = (
            self.agg_value
        )

        if packets:
            features["flow_id"] = str(get_flow_key(packets[0]))
        else:
            features["flow_id"] = "unknown"
        # ==========================

        if self.label is not None:
            features["label"] = self.label

        return features

    # ------------------------------------------------------------------
    # Compatibility layer for ONLINE mode
    # ------------------------------------------------------------------

    def _calculate_features(self, packets):
        """
        Backward compatibility wrapper.

        Your old main.py calls:
            extractor._calculate_features(...)

        so we keep this method.
        """

        return self.calculate_features(
            packets
        )

    # ------------------------------------------------------------------
    # Offline PCAP -> CSV extraction
    # ------------------------------------------------------------------

    def process_and_save(
            self,
            output_csv,
            batch_size=10000
    ):

        if self.pcap_path is None:
            raise ValueError(
                "pcap_path cannot be None "
                "during offline extraction."
            )

        processor = OfflineProcessor(
            extractor=self
        )

        processor.process_and_save(
            output_csv=output_csv,
            batch_size=batch_size
        )