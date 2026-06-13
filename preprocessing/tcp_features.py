from scapy.layers.inet import TCP

TCP_FLAGS = {
    "SYN": 0x02,
    "ACK": 0x10,
    "FIN": 0x01,
    "RST": 0x04,
    "PSH": 0x08,
    "URG": 0x20,
}


def tcp_flag_counts(packets):
    counts = {flag: 0 for flag in TCP_FLAGS}

    for packet in packets:

        if not packet.haslayer(TCP):
            continue

        flags = packet[TCP].flags

        for flag_name, mask in TCP_FLAGS.items():
            if flags & mask:
                counts[flag_name] += 1

    return counts


def retransmission_count(packets):
    seen = set()
    retransmissions = 0

    for packet in packets:

        if not packet.haslayer(TCP):
            continue

        seq = packet[TCP].seq

        if seq in seen:
            retransmissions += 1
        else:
            seen.add(seq)

    return retransmissions
