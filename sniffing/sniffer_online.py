import queue
import time 
import threading
from scapy.all import sniff
from preprocessing.feature_extractor import FlowFeatureExtractor 

class SnifferOnline:
    def __init__(self, interface=None, agg_mode="packet", agg_value=50, prediction_callback=None):
        self.interface = interface
        self.agg_mode = agg_mode
        self.agg_value = agg_value
        self.prediction_callback = prediction_callback
        
        self.stop_sniffing = False
        self.packet_queue = queue.Queue(maxsize=15000) 
        self.dropped_packets = 0
        self.last_drop_warning = time.time()
        
        self.active_flows = {}
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

    def _flow_processing_worker(self):
        """Background thread to consume packets and generate feature vectors."""
        print(f"[*] Online Worker started. Mode: {self.agg_mode}, Value: {self.agg_value}")
        
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
                    if len(flow_packets) >= self.agg_value:
                        flush_flow = True
                        
                elif self.agg_mode == "time":
                    time_diff = float(flow_packets[-1].time - flow_packets[0].time)
                    if time_diff >= self.agg_value:
                        flush_flow = True

                if flush_flow:
                    features = self.feature_extractor._calculate_features(flow_packets)
                    self.active_flows[flow_key] = [] 
                    
                    if self.prediction_callback:
                        self.prediction_callback(features, flow_key)
                    else:
                        print(f"[+] New Flow Aggregate Ready -> Features extracted: {len(features)}")
                    
            except queue.Empty:
                pass

    def start_capture(self):
        print(f"[*] Starting online capture on interface: {self.interface or 'default'}")
        
        self.worker_thread = threading.Thread(target=self._flow_processing_worker, daemon=True)
        self.worker_thread.start()
        
        sniff(
            iface=self.interface,
            filter="ip or ip6",
            prn=self._packet_handler,
            store=False,
            stop_filter=lambda x: self.stop_sniffing
        )
        
    def stop_capture(self):
        self.stop_sniffing = True
        try:
            self.packet_queue.put(None, timeout=2)
        except queue.Full:
            pass
        print("[*] Capture stopped.")