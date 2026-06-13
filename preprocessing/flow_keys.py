# preprocessing/flow_keys.py
from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.inet6 import IPv6

def get_flow_key(packet):
    """
    Direction-agnostic key (Canonical Key).
    Dzięki temu pakiety A -> B oraz B -> A trafiają do tego samego przepływu,
    co pozwala policzyć upload i download (fwd/bwd ratio).
    """
    if packet.haslayer(IP):
        ip_layer = packet[IP]
        proto = ip_layer.proto
    elif packet.haslayer(IPv6):
        ip_layer = packet[IPv6]
        proto = ip_layer.nh
    else:
        return None

    if packet.haslayer(TCP):
        sport, dport = packet[TCP].sport, packet[TCP].dport
    elif packet.haslayer(UDP):
        sport, dport = packet[UDP].sport, packet[UDP].dport
    else:
        return None

    src_ip, dst_ip = ip_layer.src, ip_layer.dst
    
    # Sortowanie gwarantuje, że niezależnie od kierunku, klucz będzie taki sam
    if (src_ip, sport) > (dst_ip, dport):
        return (dst_ip, src_ip, dport, sport, proto)
    
    return (src_ip, dst_ip, sport, dport, proto)