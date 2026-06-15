import queue
import time
import socket
from scapy.all import get_if_list, get_if_addr
from scapy.sendrecv import sniff
from scapy.layers.inet import IP, UDP, TCP
from scapy.layers.inet6 import IPv6
from preprocessing.feature_extractor import FlowFeatureExtractor

# Porty które nie mają sensu klasyfikować (broadcast/discovery/mDNS)
_NOISE_PORTS = {5353, 137, 138, 1900, 5355, 6666, 12177}

# Prefiksy adresów multicast/broadcast
_MULTICAST_PREFIXES = ("224.", "239.", "255.", "ff02", "ff05")


def _is_noise_packet(packet) -> bool:
    """Zwraca True jeśli pakiet to ruch sieciowy który nie powinien być klasyfikowany."""
    if packet.haslayer(IP):
        dst = packet[IP].dst
        if any(dst.startswith(p) for p in _MULTICAST_PREFIXES):
            return True
        if dst == "255.255.255.255":
            return True
    elif packet.haslayer(IPv6):
        dst = packet[IPv6].dst
        if dst.startswith("ff"):
            return True

    if packet.haslayer(UDP):
        if packet[UDP].dport in _NOISE_PORTS or packet[UDP].sport in _NOISE_PORTS:
            return True

    return False


class SnifferOnline:
    """
    mode="flow": agreguje pakiety w przepływy (jak RF) i wywołuje
                 prediction_callback(features, flow_key) po flushu.
    mode="raw":  każdy pakiet trafia bezpośrednio do
                 packet_callback(packet) (per-pakiet, jak CNN).
    """

    # Czas (w sekundach) po którym flow pakietowy zostaje wymuszony,
    # nawet jeśli nie zebrał wymaganej liczby pakietów.
    PACKET_FLOW_TIMEOUT = 5.0

    # Minimalna liczba pakietów w flow żeby go w ogóle klasyfikować (timeout flush).
    MIN_PKTS_FOR_TIMEOUT_FLUSH = 3

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
        # Śledzenie czasu ostatniego pakietu per flow (do timeout flush)
        self.flow_last_seen = {}

        self.feature_extractor = FlowFeatureExtractor(
            agg_mode=self.agg_mode,
            agg_value=self.agg_value
        )

    def _packet_handler(self, packet):
        """Wywoływane przez sniff() dla każdego pakietu — tylko kolejkuje."""
        # Odfiltruj szum sieciowy zanim trafi do kolejki
        if _is_noise_packet(packet):
            return

        try:
            self.packet_queue.put(packet, block=False)
        except queue.Full:
            self.dropped_packets += 1
            current_time = time.time()
            if current_time - self.last_drop_warning > 5.0:
                print(f"\n[!] Warning: Dropped {self.dropped_packets} packets. "
                      "Consider increasing throughput or reducing agg_value.")
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
                if get_if_addr(iface) == local_ip:
                    print(f"[*] Wykryto aktywną kartę: {iface} ({local_ip})")
                    return iface
            except Exception:
                continue
        return None

    # ------------------------------------------------------------------
    # MODE "raw" - per-pakiet (CNN)
    # ------------------------------------------------------------------
    def _raw_packet_worker(self):
        print("[*] Online Worker started (raw per-packet mode with BATCHING).")
        batch = []
        batch_size = 32 # Ilość pakietów paczkowanych do jednej operacji (zwiększa FPS wielokrotnie)
        
        while not self.stop_sniffing:
            try:
                # Pobieramy bardzo szybko
                packet = self.packet_queue.get(timeout=0.05)
                if packet is not None:
                    batch.append(packet)
                self.packet_queue.task_done()
            except queue.Empty:
                pass
            
            # Puszczamy inferencję gdy: uzbieraliśmy pełny batch ALBO kolejka jest pusta a mamy coś w buforze
            if len(batch) >= batch_size or (len(batch) > 0 and self.packet_queue.empty()):
                self.packet_callback(batch)  # <--- UWAGA: Wysyłamy teraz LISTĘ pakietów do dashboardu!
                batch = []

    # ------------------------------------------------------------------
    # MODE "flow" - agregacja przepływów (RF)
    # ------------------------------------------------------------------
    def _flow_processing_worker(self):
        print(f"[*] Online Worker started (flow mode). "
              f"Mode: {self.agg_mode}, Value: {self.agg_value}")

        while not self.stop_sniffing:
            try:
                packet = self.packet_queue.get(timeout=1.0)
                if packet is None:
                    continue

                flow_key = self.feature_extractor._get_flow_key(packet)
                if not flow_key:
                    self.packet_queue.task_done()
                    continue

                now = time.time()

                if flow_key not in self.active_flows:
                    self.active_flows[flow_key] = []

                self.active_flows[flow_key].append(packet)
                self.flow_last_seen[flow_key] = now
                flow_packets = self.active_flows[flow_key]

                flush_flow = False
                if self.agg_mode == "packet":
                    flush_flow = len(flow_packets) >= self.agg_value
                elif self.agg_mode == "time":
                    time_diff = float(flow_packets[-1].time - flow_packets[0].time)
                    flush_flow = time_diff >= self.agg_value

                if flush_flow:
                    self._flush_flow(flow_key, flow_packets)

                self.packet_queue.task_done()

            except queue.Empty:
                # Kolejka pusta — dobry moment żeby sprawdzić stare flow
                self._timeout_flush_stale_flows()

    def _flush_flow(self, flow_key, flow_packets):
        """Klasyfikuje flow i czyści bufor."""
        features = self.feature_extractor._calculate_features(flow_packets)
        self.active_flows[flow_key] = []
        self.flow_last_seen.pop(flow_key, None)
        self.prediction_callback(features, flow_key)

    def _timeout_flush_stale_flows(self):
        """
        Wymusza klasyfikację flowów które nie dostały nowego pakietu
        od PACKET_FLOW_TIMEOUT sekund.

        Działa zarówno dla trybu 'packet' jak i 'time' — w obu przypadkach
        flow który "stoi" jest lepiej sklasyfikować na podstawie tego co mamy
        niż czekać w nieskończoność.
        """
        now = time.time()
        keys_to_flush = []

        for key, last_seen in list(self.flow_last_seen.items()):
            if now - last_seen < self.PACKET_FLOW_TIMEOUT:
                continue

            pkts = self.active_flows.get(key, [])
            if len(pkts) >= self.MIN_PKTS_FOR_TIMEOUT_FLUSH:
                keys_to_flush.append(key)
            elif pkts:
                # Za mało pakietów — po prostu wyczyść bufor bez klasyfikacji
                self.active_flows[key] = []
                self.flow_last_seen.pop(key, None)

        for key in keys_to_flush:
            pkts = self.active_flows.get(key, [])
            if pkts:
                self._flush_flow(key, pkts)

    # ------------------------------------------------------------------
    def start_capture(self):
        if not self.interface:
            self.interface = self._detect_active_iface()

        if not self.interface:
            print("[!] Nie udało się wykryć aktywnej karty, używam domyślnej.")

        print(f"[*] Starting capture on: {self.interface}")

        # Uruchamiamy worker PRZED sniff() — sniff() blokuje wątek
        if self.mode == "flow":
            worker_fn = self._flow_processing_worker
        else:
            worker_fn = self._raw_packet_worker

        worker_thread = threading.Thread(target=worker_fn, daemon=True)
        worker_thread.start()

        sniff(
            prn=self._packet_handler,
            store=False,
            iface=self.interface,
            promisc=True,
            stop_filter=lambda _: self.stop_sniffing,
        )

    def stop_capture(self):
        self.stop_sniffing = True
        try:
            self.packet_queue.put(None, timeout=2)
        except queue.Full:
            pass
        print("[*] Capture stopped.")


import threading