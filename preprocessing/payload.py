from scapy.layers.inet import TCP, UDP


def get_payload_length(packet):
    if packet.haslayer(TCP) and packet[TCP].payload:
        return len(packet[TCP].payload)

    if packet.haslayer(UDP) and packet[UDP].payload:
        return len(packet[UDP].payload)

    return 0
