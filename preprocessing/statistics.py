import numpy as np

from preprocessing.payload import get_payload_length

EMPTY_DIRECTION_FEATURES = {
    "pkt_count": 0.0,
    "pkt_len_mean": 0.0,
    "pkt_len_std": 0.0,
    "pkt_len_p50": 0.0,
    "pkt_len_p75": 0.0,
    "pkt_len_max": 0.0,
    "payload_median": 0.0,
    "iat_mean": 0.0,
    "iat_std": 0.0,
    "iat_median": 0.0,
    "iat_cov": 0.0,
}


def direction_features(packets, prefix):
    if not packets:
        return {
            f"{prefix}_{k}": v
            for k, v in EMPTY_DIRECTION_FEATURES.items()
        }

    packet_lengths = [len(packet) for packet in packets]

    payload_lengths = [
        get_payload_length(packet)
        for packet in packets
    ]

    iats = [
        float(packets[i].time - packets[i - 1].time)
        for i in range(1, len(packets))
    ]

    if not iats:
        iats = [0.0]

    iat_mean = float(np.mean(iats))
    iat_std = float(np.std(iats))

    return {
        f"{prefix}_pkt_count": len(packets),

        f"{prefix}_pkt_len_mean": float(np.mean(packet_lengths)),

        f"{prefix}_pkt_len_std": float(np.std(packet_lengths)),

        f"{prefix}_pkt_len_p50": float(np.percentile(packet_lengths, 50)),

        f"{prefix}_pkt_len_p75": float(np.percentile(packet_lengths, 75)),

        f"{prefix}_pkt_len_max": float(np.max(packet_lengths)),

        f"{prefix}_payload_median": float(np.median(payload_lengths)),

        f"{prefix}_iat_mean": iat_mean,

        f"{prefix}_iat_std": iat_std,

        f"{prefix}_iat_median": float(np.median(iats)),

        f"{prefix}_iat_cov": (
            iat_std / iat_mean
            if iat_mean > 0 else 0.0
        )
    }
