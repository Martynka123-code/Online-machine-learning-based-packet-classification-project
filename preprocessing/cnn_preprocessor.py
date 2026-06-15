# preprocessing/cnn_preprocessor.py
import numpy as np
from scapy.all import raw
from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.inet6 import IPv6
from scapy.utils import PcapReader

# ── 1. DEFINICJE DO FILTROWANIA SZUMU (Rozwiązanie Problemu 3) ──
_NOISE_PORTS = {5353, 137, 138, 1900, 5355, 6666, 12177}
_MULTICAST_PREFIXES = ("224.", "239.", "255.", "ff02", "ff05")

def is_noise_packet(packet) -> bool:
    if packet.haslayer(IP):
        dst = packet[IP].dst
        if any(dst.startswith(p) for p in _MULTICAST_PREFIXES) or dst == "255.255.255.255":
            return True
    elif packet.haslayer(IPv6):
        if packet[IPv6].dst.startswith("ff"):
            return True
    if packet.haslayer(UDP):
        if packet[UDP].dport in _NOISE_PORTS or packet[UDP].sport in _NOISE_PORTS:
            return True
    return False


class CNNPreprocessor:
    def __init__(self, max_length=1000, local_ips=None):
        """
        local_ips: zbiór lokalnych adresów IP (konieczne do kodowania kierunku)
        """
        self.max_length = max_length
        self.data = []
        self.labels = []
        self.local_ips = local_ips or set()

    def packet_to_bytes(self, packet):
        # ── 2. ODRZUCENIE L2 (Rozwiązanie Problemu 2 - Alignment Shift) ──
        if packet.haslayer(IP):
            l3_packet = packet[IP]
        elif packet.haslayer(IPv6):
            l3_packet = packet[IPv6]
        else:
            return None # Ignorujemy pakiety bez warstwy sieciowej
            
        # Klonujemy, żeby nie zepsuć oryginalnego pakietu w systemie online
        pkt = l3_packet.copy()

        # ── 3. KODOWANIE KIERUNKU (Rozwiązanie Problemu 4 - Directionality) ──
        if pkt.src in self.local_ips:
            # Ruch wychodzący (fwd) -> my do serwera
            pkt.src = "1.1.1.1"
            pkt.dst = "2.2.2.2"
        else:
            # Ruch przychodzący (bwd) -> serwer do nas
            pkt.src = "2.2.2.2"
            pkt.dst = "1.1.1.1"

        # ── 4. CZYSZCZENIE WYCIEKÓW DANYCH (Rozwiązanie Problemu 1 - Data Leaks) ──
        if pkt.haslayer(IP):
            pkt[IP].id = 0
            pkt[IP].chksum = 0
        elif pkt.haslayer(IPv6):
            pkt[IPv6].fl = 0 # Flow label
            
        if pkt.haslayer(TCP):
            pkt[TCP].seq = 0
            pkt[TCP].ack = 0
            pkt[TCP].chksum = 0
            pkt[TCP].options = [] # Usuwamy timestampy
            # Maskujemy porty efemeryczne (losowe porty klienta), zostawiamy porty usług (np. 443)
            if pkt[TCP].sport > 32768: pkt[TCP].sport = 0
            if pkt[TCP].dport > 32768: pkt[TCP].dport = 0
            
        elif pkt.haslayer(UDP):
            pkt[UDP].chksum = 0
            if pkt[UDP].sport > 32768: pkt[UDP].sport = 0
            if pkt[UDP].dport > 32768: pkt[UDP].dport = 0
            
        # Konwersja i Normalizacja
        raw_bytes = bytearray(raw(pkt))
        
        if len(raw_bytes) > self.max_length:
            raw_bytes = raw_bytes[:self.max_length]
        else:
            raw_bytes += bytearray([0] * (self.max_length - len(raw_bytes)))
            
        return np.array(raw_bytes, dtype=np.float32) / 255.0

    def process_pcap(self, pcap_path, label_idx):
        print(f"[*] Processing PCAP for CNN: {pcap_path} | Class ID: {label_idx}")
        
        with PcapReader(pcap_path) as packets:
            for pkt in packets:
                if not (pkt.haslayer(TCP) or pkt.haslayer(UDP)):
                    continue
                
                # Zastosowanie tego samego filtra szumów co w online!
                if is_noise_packet(pkt):
                    continue
                    
                byte_array = self.packet_to_bytes(pkt)
                if byte_array is not None:
                    self.data.append(byte_array)
                    self.labels.append(label_idx)

    def save_dataset(self, output_path):
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