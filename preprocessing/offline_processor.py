import os
import pandas as pd

from scapy.utils import PcapReader

from preprocessing.flow_keys import get_flow_key


class OfflineProcessor:
    """
    Handles offline PCAP processing.

    Responsibilities:
    - read packets from PCAP
    - aggregate packets into flows
    - flush flows according to aggregation mode
    - save extracted features into CSV
    """

    def __init__(self, extractor):

        self.extractor = extractor

        self.flows = {}

        self.dataset = []

    # ------------------------------------------------------------------
    # Main extraction loop
    # ------------------------------------------------------------------

    def process_and_save(
            self,
            output_csv,
            batch_size=10000
    ):

        print(
            f"[*] Extracting: "
            f"{self.extractor.pcap_path} "
            f"| Mode: {self.extractor.agg_mode} "
            f"| Value: {self.extractor.agg_value}"
        )

        with PcapReader(
                self.extractor.pcap_path
        ) as packets:

            for packet in packets:

                flow_key = get_flow_key(
                    packet
                )

                if flow_key is None:
                    continue

                # ------------------------------------------------------
                # Create flow buffer
                # ------------------------------------------------------

                if flow_key not in self.flows:
                    self.flows[flow_key] = []

                self.flows[flow_key].append(
                    packet
                )

                flow_packets = self.flows[
                    flow_key
                ]

                # ------------------------------------------------------
                # Decide if flow should be flushed
                # ------------------------------------------------------

                flush_flow = False

                # ---------------------------
                # Packet-based aggregation
                # ---------------------------

                if (
                        self.extractor.agg_mode
                        == "packet"
                ):

                    flush_flow = (
                            len(flow_packets)
                            >= self.extractor.agg_value
                    )

                # ---------------------------
                # Time-based aggregation
                # ---------------------------

                elif (
                        self.extractor.agg_mode
                        == "time"
                ):

                    duration = float(
                        flow_packets[-1].time
                        - flow_packets[0].time
                    )

                    flush_flow = (
                            duration
                            >= self.extractor.agg_value
                    )

                # ------------------------------------------------------
                # Extract and flush features
                # ------------------------------------------------------

                if flush_flow:

                    features = (
                        self.extractor
                        .calculate_features(
                            flow_packets
                        )
                    )

                    self.dataset.append(
                        features
                    )

                    # Reset flow buffer
                    self.flows[flow_key] = []

                    # Save batch
                    if (
                            len(self.dataset)
                            >= batch_size
                    ):
                        self._save_batch(
                            output_csv
                        )

        # --------------------------------------------------------------
        # Flush remaining unfinished flows
        # --------------------------------------------------------------

        self._flush_remaining_flows()

        # --------------------------------------------------------------
        # Save final batch
        # --------------------------------------------------------------

        if self.dataset:
            self._save_batch(
                output_csv
            )

    # ------------------------------------------------------------------
    # Flush remaining packets after PCAP ends
    # ------------------------------------------------------------------

    def _flush_remaining_flows(self):

        for flow_packets in self.flows.values():

            if not flow_packets:
                continue

            features = (
                self.extractor
                .calculate_features(
                    flow_packets
                )
            )

            self.dataset.append(
                features
            )

    # ------------------------------------------------------------------
    # Save dataframe batch
    # ------------------------------------------------------------------

    def _save_batch(
            self,
            output_csv
    ):

        dataframe = pd.DataFrame(
            self.dataset
        ).round(5)

        file_exists = os.path.isfile(
            output_csv
        )

        dataframe.to_csv(
            output_csv,
            mode="a",
            header=not file_exists,
            index=False
        )

        print(
            f"[+] Saved batch of "
            f"{len(self.dataset)} "
            f"aggregates -> {output_csv}"
        )

        self.dataset.clear()
