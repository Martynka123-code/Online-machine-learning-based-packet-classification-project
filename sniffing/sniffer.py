from scapy.all import sniff, PcapWriter
import os
from datetime import datetime

INTERFACE = None
DIR_FOR_SAVING = "traffic_logs"
BRF_FILTER = "ip"
CHECK_INTERVAL = 1000000

if not os.path.exists(DIR_FOR_SAVING):
    os.makedirs(DIR_FOR_SAVING)

class Sniffer:
    def __init__(self):
        self.today_date = None
        self.writer = None
        self.packet_count = 0  

    def get_file_name(self):
        date_str = datetime.now().strftime("%Y-%m-%d")
        if self.today_date != date_str:
            if self.writer:
                self.writer.close()
            self.today_date = date_str
            file_name = f"{DIR_FOR_SAVING}/traffic_{date_str}.pcap"
            self.writer = PcapWriter(file_name, append=True, sync=True)
            print(f"[*] Writing to: {file_name}")

    def packet_handler(self, packet):
        try:
            if self.packet_count % CHECK_INTERVAL == 0:
                self.get_file_name()
            
            self.packet_count += 1
            self.writer.write(packet)
        except Exception as e:
            print(f"Write error: {e}")

    def start(self):
        print(f"[*] Sniffer started. Data is being saved to folder: {DIR_FOR_SAVING}")
        print("[*] Press Ctrl+C to stop.")
        sniff(filter=BRF_FILTER, prn=self.packet_handler, store=False)

if __name__ == "__main__":
    sniffer = Sniffer()
    sniffer.start()