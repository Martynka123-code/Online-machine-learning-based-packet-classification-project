from scapy.all import sniff

class SnifferOnline:
    def __init__(self, callback_function, interface=None):
        """
        callback_function: function to which the sniffer will send each captured packet.
        interface: optional network interface name (e.g., 'eth0', 'Wi-Fi').
        """
        self.callback = callback_function
        self.interface = interface
        self.stop_sniffing = False

    def _packet_handler(self, packet):
        # Forward the packet directly to the processing pipeline
        self.callback(packet)

    def start_capture(self):
        print(f"[*] Starting online capture on interface: {self.interface or 'default'}...")
        print("[*] Press Ctrl+C to stop.")
        # store=False is critical to prevent RAM exhaustion!
        sniff(
            iface=self.interface, 
            prn=self._packet_handler, 
            store=False, 
            stop_filter=lambda x: self.stop_sniffing
        )

    def stop_capture(self):
        self.stop_sniffing = True
        print("[*] Capture stopped.")