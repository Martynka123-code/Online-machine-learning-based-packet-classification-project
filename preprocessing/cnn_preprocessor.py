# preprocessing/cnn_preprocessor.py
import os
import numpy as np
from scapy.all import raw
from scapy.layers.l2 import Ether
from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.inet6 import IPv6
from scapy.utils import PcapReader

class CNNPreprocessor:
    def __init__(self, max_length=1000):
        """
        max_length: Target number of bytes per packet (must match signal_length in the CNN model)
        """
        self.max_length = max_length
        self.data = []
        self.labels = []

    def packet_to_bytes(self, packet):
        """Masks the packet, converts it to bytes, truncates/pads to max_length, and normalizes."""
        
        if packet.haslayer(Ether):
            packet[Ether].src = "00:00:00:00:00:00"
            packet[Ether].dst = "00:00:00:00:00:00"
            
        if packet.haslayer(IP):
            packet[IP].src = "0.0.0.0"
            packet[IP].dst = "0.0.0.0"
            packet[IP].chksum = 0
        elif packet.haslayer(IPv6):
            packet[IPv6].src = "::"
            packet[IPv6].dst = "::"
            
        if packet.haslayer(TCP):
            packet[TCP].chksum = 0
        elif packet.haslayer(UDP):
            packet[UDP].chksum = 0
            
        raw_bytes = bytearray(raw(packet))
        
        if len(raw_bytes) > self.max_length:
            raw_bytes = raw_bytes[:self.max_length]
        else:
            raw_bytes += bytearray([0] * (self.max_length - len(raw_bytes)))
            
        normalized_array = np.array(raw_bytes, dtype=np.float32) / 255.0
        
        return normalized_array

    def process_pcap(self, pcap_path, label_idx):
        """Reads a PCAP file and converts each packet into a normalized vector."""
        print(f"[*] Processing PCAP for CNN: {pcap_path} | Class ID: {label_idx}")
        
        with PcapReader(pcap_path) as packets:
            for pkt in packets:
                if not (pkt.haslayer(TCP) or pkt.haslayer(UDP)):
                    continue
                    
                byte_array = self.packet_to_bytes(pkt)
                self.data.append(byte_array)
                self.labels.append(label_idx)

    def save_dataset(self, output_path):
        """Saves the collected tensors to a compressed .npz file."""
        if not self.data:
            print("[!] No data to save.")
            return
            
        np.savez_compressed(
            output_path, 
            features=np.array(self.data, dtype=np.float32), 
            labels=np.array(self.labels, dtype=np.int64)
        )
        print(f"[+] Saved CNN dataset to {output_path} (Samples: {len(self.data)})")
        
        self.data.clear()
        self.labels.clear()
        