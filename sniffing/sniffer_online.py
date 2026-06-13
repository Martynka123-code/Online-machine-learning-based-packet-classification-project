import queue
import time
import socket
from scapy.all import get_if_list, get_if_addr
from scapy.sendrecv import sniff
from preprocessing.feature_extractor import FlowFeatureExtractor


class SnifferOnline:
    """
    mode="flow": agreguje pakiety w przepływy (jak RF) i wywołuje
                 prediction_callback(features, flow_key) po flushu.
    mode="raw":  każdy pakiet trafia bezpośrednio do
                 packet_callback(packet) (per-pakiet, jak CNN).

    Tylko JEDEN wątek konsumuje packet_queue - wybrany przez `mode`.
    """

    def __init__(self, interface=None, agg_mode="packet", agg_value=50,
                 mode="flow", prediction_callback=None, packet_callback=None):

        if mode not in ("flow", "raw"):
            raise ValueError("mode musi być 'flow' albo 'raw'")
        if mode == "flow" and prediction_callback is None:
            raise ValueError("mode='flow' wymaga prediction_callback")
        if mode == "raw" and packet_callback is None:
            raise ValueError("mode='raw' wymaga packet_callback")

        self.interface = interface
        self.agg_mode = agg_mode
        self.agg_value = agg_value
        self.mode = mode
        self.prediction_callback = prediction_callback
        self.packet_callback = packet_callback

        self.stop_sniffing = False
        self.packet_queue = queue.Queue(maxsize=15000)
        self.dropped_packets = 0
        self.last_drop_warning = time.time()

        self.active_flows = {}
        # Potrzebny tylko w mode="flow", ale konstrukcja jest darmowa
        self.feature_extractor = FlowFeatureExtractor(agg_mode=self.agg_mode, agg_value=self.agg_value)

    def _packet_handler(self, packet):
        try:
            self.packet_queue.put(packet, block=False)
        except queue.Full:
            self.dropped_packets += 1
            current_time = time.time()
            if current_time - self.last_drop_warning > 5.0:
                print(f"\n[!] Warning: Dropped {self.dropped_packets} packets. "
                      "Consider increasing throughput (e.g., smaller aggregation value).")
                self.last_drop_warning = current_time

    def _detect_active_iface(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        try:
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
        finally:
            s.close()

        for iface in get_if_list():
            try:
                iface_ip = get_if_addr(iface)

                if iface_ip == local_ip:
                    print(f"[*] Wykryto aktywną kartę: {iface} ({iface_ip})")
                    return iface

            except Exception:
                continue

        return None

    # ------------------------------------------------------------------
    # MODE "raw" - per-pakiet (CNN)
    # ------------------------------------------------------------------
    def _raw_packet_worker(self):
        print("[*] Online Worker started (raw per-packet mode).")
        while not self.stop_sniffing:
            try:
                packet = self.packet_queue.get(timeout=1.0)
                if packet is None:
                    continue
                self.packet_callback(packet)
                self.packet_queue.task_done()
            except queue.Empty:
                continue

    # ------------------------------------------------------------------
    # MODE "flow" - agregacja przepływów (RF)
    # ------------------------------------------------------------------
    def _flow_processing_worker(self):
        print(f"[*] Online Worker started (flow mode). Mode: {self.agg_mode}, Value: {self.agg_value}")

        while not self.stop_sniffing:
            try:
                packet = self.packet_queue.get(timeout=1.0)
                if packet is None:
                    continue

                flow_key = self.feature_extractor._get_flow_key(packet)
                if not flow_key:
                    continue

                if flow_key not in self.active_flows:
                    self.active_flows[flow_key] = []

                self.active_flows[flow_key].append(packet)
                flow_packets = self.active_flows[flow_key]

                flush_flow = False
                if self.agg_mode == "packet":
                    flush_flow = len(flow_packets) >= self.agg_value
                elif self.agg_mode == "time":
                    time_diff = float(flow_packets[-1].time - flow_packets[0].time)
                    flush_flow = time_diff >= self.agg_value

                if flush_flow:
                    features = self.feature_extractor._calculate_features(flow_packets)
                    self.active_flows[flow_key] = []
                    self.prediction_callback(features, flow_key)

                self.packet_queue.task_done()

            except queue.Empty:
                if self.agg_mode == "time":
                    keys_to_flush = []
                    current_time = time.time() 
                    
                    for key, pkts in self.active_flows.items():
                        if len(pkts) < 2:
                            continue
                        
                        duration = current_time - float(pkts[0].time)
                        if duration >= self.agg_value:
                            keys_to_flush.append(key)

                    for key in keys_to_flush:
                        flow_packets = self.active_flows[key]
                        features = self.feature_extractor._calculate_features(flow_packets)
                        self.active_flows[key] = []
                        self.prediction_callback(features, key)

    # ------------------------------------------------------------------
    def start_capture(self):

        if not self.interface:
            self.interface = self._detect_active_iface()

        if not self.interface:
            print("[!] Nie udało się wykryć aktywnej karty, używam domyślnej.")

        print(f"[*] Starting capture on: {self.interface}")

        sniff(
            prn=self._packet_handler,
            store=False,
            iface=self.interface,
            promisc=True
        )

    def stop_capture(self):
        self.stop_sniffing = True
        try:
            self.packet_queue.put(None, timeout=2)
        except queue.Full:
            pass
        print("[*] Capture stopped.")