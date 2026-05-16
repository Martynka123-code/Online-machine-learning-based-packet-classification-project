import queue
import time 
from scapy.all import sniff

class SnifferOnline:
    def __init__(self, interface=None):
        self.interface = interface
        self.stop_sniffing = False
        self.packet_queue = queue.Queue(maxsize=15000) 
        self.dropped_packets = 0
        self.last_drop_warning = time.time()

    def _packet_handler(self, packet):
        try:
            self.packet_queue.put(packet, block=False)
        except queue.Full:
            self.dropped_packets += 1
            current_time = time.time()
            if current_time - self.last_drop_warning > 5.0:
                print(f"\n[!] Warning: Dropped {self.dropped_packets} packets. " 
                      "Consider increasing throughput (e.g., smaller granularity).")
                self.last_drop_warning = current_time
    def start_capture(self):
        print(f"[*] Starting online capture on interface: {self.interface or 'default'}")
        sniff(
            iface=self.interface,
            filter="ip or ipv6",
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