import queue
from scapy.all import sniff

class SnifferOnline:
    def __init__(self, interface=None):
        self.interface = interface
        self.stop_sniffing = False
        self.packet_queue = queue.Queue(maxsize=15000) 

    def _packet_handler(self, packet):
        try:
            self.packet_queue.put(packet, block=False)
        except queue.Full:
            pass 
    def start_capture(self):
        print(f"[*] Starting online capture on interface: {self.interface or 'default'}")
        sniff(
            iface=self.interface,
            filter="ip",
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