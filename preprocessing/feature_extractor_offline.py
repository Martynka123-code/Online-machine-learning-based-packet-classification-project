import os
import numpy as np
import pandas as pd
from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.inet6 import IPv6
from scapy.utils import PcapReader
import warnings

warnings.filterwarnings('ignore')


class FlowFeatureExtractorOffline:
    def __init__(self, pcap_path, label, agg_mode="packet", agg_value=100):
        """
        Offline extractor — reads from a PCAP file and saves to CSV.

        agg_mode : "packet" – flush window every N packets
                   "time"   – flush window every T seconds
        agg_value: threshold for the chosen mode
        """
        self.pcap_path = pcap_path
        self.label = label
        self.agg_mode = agg_mode
        self.agg_value = agg_value
        self.flows = {}
        self.dataset = []

    # ------------------------------------------------------------------
    # Flow key helpers
    # ------------------------------------------------------------------

    def _get_flow_key(self, packet):
        """Direction-aware 5-tuple (src_ip, dst_ip, sport, dport, proto)."""
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

        return (ip_layer.src, ip_layer.dst, sport, dport, proto)

    def _get_canonical_key(self, packet):
        """Direction-agnostic key — groups both directions of the same flow."""
        key = self._get_flow_key(packet)
        if key is None:
            return None
        src_ip, dst_ip, sport, dport, proto = key
        if (src_ip, sport) > (dst_ip, dport):
            return (dst_ip, src_ip, dport, sport, proto)
        return key

    # ------------------------------------------------------------------
    # Payload
    # ------------------------------------------------------------------

    def _get_payload_length(self, packet):
        if packet.haslayer(TCP) and packet[TCP].payload:
            return len(packet[TCP].payload)
        elif packet.haslayer(UDP) and packet[UDP].payload:
            return len(packet[UDP].payload)
        return 0

    # ------------------------------------------------------------------
    # Direction split
    # ------------------------------------------------------------------

    def _split_by_direction(self, packets):
        """Split packets into fwd (initiator→responder) and bwd (responder→initiator)."""
        if not packets:
            return [], []
        first_key = self._get_flow_key(packets[0])
        if first_key is None:
            return packets, []
        initiator_ip = first_key[0]
        fwd, bwd = [], []
        for p in packets:
            k = self._get_flow_key(p)
            if k is None:
                continue
            (fwd if k[0] == initiator_ip else bwd).append(p)
        return fwd, bwd

    # ------------------------------------------------------------------
    # TCP helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tcp_flag_counts(packets):
        counts = {"SYN": 0, "ACK": 0, "FIN": 0, "RST": 0, "PSH": 0, "URG": 0}
        for p in packets:
            if p.haslayer(TCP):
                f = p[TCP].flags
                if f & 0x02: counts["SYN"] += 1
                if f & 0x10: counts["ACK"] += 1
                if f & 0x01: counts["FIN"] += 1
                if f & 0x04: counts["RST"] += 1
                if f & 0x08: counts["PSH"] += 1
                if f & 0x20: counts["URG"] += 1
        return counts

    @staticmethod
    def _retransmission_count(packets):
        seen, retrans = set(), 0
        for p in packets:
            if p.haslayer(TCP):
                seq = p[TCP].seq
                if seq in seen:
                    retrans += 1
                else:
                    seen.add(seq)
        return retrans

    # ------------------------------------------------------------------
    # Per-direction stats
    # ------------------------------------------------------------------

    def _direction_features(self, packets, prefix):
        if not packets:
            return {k: 0.0 for k in [
                f"{prefix}_pkt_count", f"{prefix}_pkt_len_mean", f"{prefix}_pkt_len_std",
                f"{prefix}_pkt_len_p50", f"{prefix}_pkt_len_p75", f"{prefix}_pkt_len_max",
                f"{prefix}_payload_median", f"{prefix}_iat_mean", f"{prefix}_iat_std",
                f"{prefix}_iat_median", f"{prefix}_iat_p75_minus_p50",
                f"{prefix}_iat_p95_minus_p50", f"{prefix}_iat_cov",
            ]}

        pkt_lengths = [len(p) for p in packets]
        payload_lengths = [self._get_payload_length(p) for p in packets]
        iats = [float(packets[i].time - packets[i - 1].time) for i in range(1, len(packets))]
        if not iats:
            iats = [0.0]

        iat_mean = float(np.mean(iats))
        iat_std = float(np.std(iats))

        return {
            f"{prefix}_pkt_count":         len(packets),
            f"{prefix}_pkt_len_mean":       float(np.mean(pkt_lengths)),
            f"{prefix}_pkt_len_std":        float(np.std(pkt_lengths)),
            f"{prefix}_pkt_len_p50":        float(np.percentile(pkt_lengths, 50)),
            f"{prefix}_pkt_len_p75":        float(np.percentile(pkt_lengths, 75)),
            f"{prefix}_pkt_len_max":        float(np.max(pkt_lengths)),
            f"{prefix}_payload_median":     float(np.median(payload_lengths)),
            f"{prefix}_iat_mean":           iat_mean,
            f"{prefix}_iat_std":            iat_std,
            f"{prefix}_iat_median":         float(np.median(iats)),
            f"{prefix}_iat_p75_minus_p50":  float(np.percentile(iats, 75) - np.median(iats)),
            f"{prefix}_iat_p95_minus_p50":  float(np.percentile(iats, 95) - np.median(iats)),
            f"{prefix}_iat_cov":            (iat_std / iat_mean) if iat_mean > 0 else 0.0,
        }

    # ------------------------------------------------------------------
    # Main feature calculation
    # ------------------------------------------------------------------

    def _calculate_features(self, packets):
        fwd_pkts, bwd_pkts = self._split_by_direction(packets)
        features = {}

        features.update(self._direction_features(fwd_pkts, "fwd"))
        features.update(self._direction_features(bwd_pkts, "bwd"))

        # Upload / download ratio
        fwd_bytes = sum(len(p) for p in fwd_pkts)
        bwd_bytes = sum(len(p) for p in bwd_pkts)
        total_bytes = fwd_bytes + bwd_bytes
        features["bytes_ratio_fwd"] = (fwd_bytes / total_bytes) if total_bytes > 0 else 0.5
        features["pkt_ratio_fwd"]   = (len(fwd_pkts) / len(packets)) if packets else 0.5

        # TCP flags & retransmissions
        flag_counts = self._tcp_flag_counts(packets)
        n = max(len(packets), 1)
        for flag, count in flag_counts.items():
            features[f"tcp_{flag.lower()}_ratio"] = count / n
        features["tcp_retrans_ratio"] = self._retransmission_count(packets) / n

        # Burst detection (gap > 100 ms = burst boundary)
        all_times = sorted(float(p.time) for p in packets)
        if len(all_times) > 1:
            gaps = [all_times[i + 1] - all_times[i] for i in range(len(all_times) - 1)]
            BURST_THRESH = 0.1
            large_gaps = [g for g in gaps if g > BURST_THRESH]
            features["burst_count"]    = len(large_gaps) + 1
            features["mean_burst_gap"] = float(np.mean(large_gaps)) if large_gaps else 0.0
        else:
            features["burst_count"]    = 1
            features["mean_burst_gap"] = 0.0

        # Metadata
        features["actual_packets_in_window"] = len(packets)
        features["agg_mode"]  = self.agg_mode
        features["agg_value"] = self.agg_value
        features["label"]     = self.label

        if packets:
            features["flow_id"] = str(self._get_canonical_key(packets[0]))
        else:
            features["flow_id"] = "unknown"
        # ==========================

        features["label"]     = self.label

        return features

    # ------------------------------------------------------------------
    # Offline extraction
    # ------------------------------------------------------------------

    def process_and_save(self, output_csv, batch_size=10_000):
        print(f"[*] Extracting: {self.pcap_path} | Mode: {self.agg_mode} | Value: {self.agg_value}")

        with PcapReader(self.pcap_path) as reader:
            for pkt in reader:
                canon_key = self._get_canonical_key(pkt)
                if canon_key is None:
                    continue

                if canon_key not in self.flows:
                    self.flows[canon_key] = []
                self.flows[canon_key].append(pkt)

                flow_packets = self.flows[canon_key]
                flush = False

                if self.agg_mode == "packet":
                    flush = len(flow_packets) >= self.agg_value
                elif self.agg_mode == "time":
                    flush = float(flow_packets[-1].time - flow_packets[0].time) >= self.agg_value

                if flush:
                    self.dataset.append(self._calculate_features(flow_packets))
                    self.flows[canon_key] = []

                    if len(self.dataset) >= batch_size:
                        self._save_batch(output_csv)

        if self.dataset:
            self._save_batch(output_csv)

    def _save_batch(self, output_csv):
        df = pd.DataFrame(self.dataset).round(5)
        file_exists = os.path.isfile(output_csv)
        df.to_csv(output_csv, mode='a', header=not file_exists, index=False)
        print(f"[+] Saved batch of {len(self.dataset)} aggregates → {output_csv}")
        self.dataset.clear()