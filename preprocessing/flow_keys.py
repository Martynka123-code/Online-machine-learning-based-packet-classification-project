from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.inet6 import IPv6


def get_flow_key(packet):
    """
    Direction-agnostic 5-tuple.
    Supports IPv4 and IPv6.
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
        sport = packet[TCP].sport
        dport = packet[TCP].dport

    elif packet.haslayer(UDP):
        sport = packet[UDP].sport
        dport = packet[UDP].dport

    else:
        return None

    ips = tuple(sorted([ip_layer.src, ip_layer.dst]))
    ports = tuple(sorted([sport, dport]))

    return (
        ips[0],
        ips[1],
        ports[0],
        ports[1],
        proto
    )
