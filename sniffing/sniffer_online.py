import queue
from scapy.all import sniff

class SnifferOnline:
    def __init__(self, interface=None):
        self.interface = interface
        self.stop_sniffing = False
        self.packet_queue = queue.Queue() 

    def _packet_handler(self, packet):
        self.packet_queue.put(packet)

    def start_capture(self):
        print(f"[*] Starting online capture on interface: {self.interface or 'default'}")
        print("[*] Packets are being buffered asynchronously.")
        sniff(
            iface=self.interface,
            filter="ip",
            prn=self._packet_handler,
            store=False,
            stop_filter=lambda x: self.stop_sniffing
        )
        
    def stop_capture(self):
        self.stop_sniffing = True
        self.packet_queue.put(None)
        print("[*] Capture stopped.")