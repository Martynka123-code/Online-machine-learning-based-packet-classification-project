from scapy.layers.inet import IP
from scapy.layers.inet6 import IPv6


def split_by_direction(packets):
    if not packets:
        return [], []

    first = packets[0]

    if first.haslayer(IP):
        initiator_ip = first[IP].src

    elif first.haslayer(IPv6):
        initiator_ip = first[IPv6].src

    else:
        return packets, []

    fwd = []
    bwd = []

    for packet in packets:

        if packet.haslayer(IP):
            src = packet[IP].src

        elif packet.haslayer(IPv6):
            src = packet[IPv6].src

        else:
            continue

        if src == initiator_ip:
            fwd.append(packet)
        else:
            bwd.append(packet)

    return fwd, bwd
