import numpy as np

BURST_THRESHOLD = 0.1


def burst_features(packets):
    timestamps = sorted(float(packet.time) for packet in packets)

    if len(timestamps) <= 1:
        return {
            "burst_count": 1,
            "mean_burst_gap": 0.0
        }

    gaps = [
        timestamps[i + 1] - timestamps[i]
        for i in range(len(timestamps) - 1)
    ]

    large_gaps = [
        gap for gap in gaps
        if gap > BURST_THRESHOLD
    ]

    return {
        "burst_count": len(large_gaps) + 1,

        "mean_burst_gap": (
            float(np.mean(large_gaps))
            if large_gaps else 0.0
        )
    }
