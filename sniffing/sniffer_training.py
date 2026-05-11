import csv
import json
import os
import socket
import threading
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import psutil
from scapy.all import PcapWriter, conf, sniff
from scapy.layers.inet import IP, TCP, UDP
from config import DATA_RAW_DIR, DATA_CSV_DIR

class SnifferTraining:
    """
    Advanced Sniffer for training data collection. 
    Maps ports to processes in real-time and extracts packet features.
    """
    def __init__(self, target_apps: list[str]):
        self.target_apps = [a.lower() for a in target_apps]
        self.local_ips = self._get_local_ips()
        self.port_map = {}
        self.flow_cache = {}
        self.writers = {}
        self.active_flows = defaultdict(lambda: {"pkt_count": 0, "bytes": 0, "start": None})
        self.stop_event = threading.Event()
        self.lock = threading.Lock()

    def _get_local_ips(self) -> set[str]:
        ips = set()
        for addrs in psutil.net_if_addrs().values():
            for addr in addrs:
                if addr.family == socket.AF_INET and not addr.address.startswith("127."):
                    ips.add(addr.address)
        return ips

    def _port_mapping_loop(self):
        """Background thread to update port-to-process mapping."""
        while not self.stop_event.is_set():
            temp_map = {}
            try:
                for conn in psutil.net_connections(kind="inet"):
                    if not conn.pid or not conn.laddr: continue
                    try:
                        name = psutil.Process(conn.pid).name().lower()
                        for app in self.target_apps:
                            if app in name:
                                temp_map[conn.laddr.port] = app
                                break
                    except (psutil.NoSuchProcess, psutil.AccessDenied): continue
            except psutil.AccessDenied: pass
            
            with self.lock:
                self.port_map = temp_map
            time.sleep(1)

    def _get_flow_key(self, si, di, sp, dp, pr):
        a, b = (si, sp), (di, dp)
        return (min(a, b), max(a, b), pr)

    def _handle_packet(self, packet):
        if not packet.haslayer(IP) or not (packet.haslayer(TCP) or packet.haslayer(UDP)):
            return

        ip = packet[IP]
        layer = packet[TCP] if packet.haslayer(TCP) else packet[UDP]
        proto = "TCP" if packet.haslayer(TCP) else "UDP"
        
        src_ip, dst_ip = ip.src, ip.dst
        sport, dport = layer.sport, layer.dport
        
        # Check cache or update from port map
        key = self._get_flow_key(src_ip, dst_ip, sport, dport, proto)
        app = self.flow_cache.get(key)

        if not app:
            local_port = sport if src_ip in self.local_ips else dport
            with self.lock:
                app = self.port_map.get(local_port)
            if app:
                self.flow_cache[key] = app

        if app:
            self._save_raw_pcap(app, packet)
            # You can add CSV/JSONL recording here if needed

    def _save_raw_pcap(self, app, packet):
        date_str = datetime.now().strftime("%Y-%m-%d")
        path = os.path.join(DATA_RAW_DIR, f"{app}_{date_str}.pcap")
        
        with self.lock:
            if app not in self.writers:
                self.writers[app] = PcapWriter(path, append=True, sync=True)
            self.writers[app].write(packet)

    def start(self):
        print(f"[*] Monitoring apps: {self.target_apps}")
        threading.Thread(target=self._port_mapping_loop, daemon=True).start()
        
        print("[*] Starting capture. Press Ctrl+C to stop.")
        try:
            sniff(
                filter="ip",
                prn=self._handle_packet,
                store=False,
                stop_filter=lambda _: self.stop_event.is_set()
            )
        except KeyboardInterrupt:
            self.stop_event.set()
            print("\n[*] Stopping capture...")
        finally:
            with self.lock:
                for w in self.writers.values():
                    w.close()
            print("[*] All files saved to", DATA_RAW_DIR)