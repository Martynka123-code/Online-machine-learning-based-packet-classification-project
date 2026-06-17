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
from scapy.all import PcapWriter, conf, sniff, get_if_list, get_if_addr
from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.inet6 import IPv6
from config import DATA_RAW_DIR, DATA_CSV_DIR

class SnifferTraining:
    """
    Advanced Sniffer for training data collection. 
    Maps ports to processes in real-time and extracts packet features.
    """
    def __init__(self, target_apps: list[str]):
        self.target_apps = [a.lower() for a in target_apps]
        self.local_ips = self._get_local_ips()
        self._debug_packet_count = 0
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
        """ Background thread to update port-to-process mapping."""
        warned_access_denied = False
        first_iteration = True

        while not self.stop_event.is_set():
            temp_map = {}
            access_denied_count = 0
            total_conns = 0
            try:
                connections = psutil.net_connections(kind="inet")
                total_conns = len(connections)
                for conn in connections:
                    if not conn.pid or not conn.laddr:
                        continue
                    try:
                        name = psutil.Process(conn.pid).name().lower()
                        for app in self.target_apps:
                            if app in name:
                                temp_map[conn.laddr.port] = app
                                break
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        access_denied_count += 1
                        continue
            except psutil.AccessDenied:
                if not warned_access_denied:
                    print("[!] AccessDenied przy psutil.net_connections() — "
                          "uruchom skrypt jako Administrator (Windows) / sudo (Linux/Mac), "
                          "inaczej mapowanie portów NIGDY nie zadziała.")
                    warned_access_denied = True

            if first_iteration:
                if total_conns == 0:
                    print("[!] psutil.net_connections() zwróciło 0 połączeń — "
                          "prawdopodobnie brak uprawnień administratora.")
                elif access_denied_count == total_conns:
                    print(f"[!] Wszystkie {total_conns} połączeń odrzucone (AccessDenied) — "
                          "potrzebne uprawnienia administratora.")
                print(f"[*] Połączenia widoczne: {total_conns} | "
                      f"Dopasowane do target_apps: {len(temp_map)}")
                first_iteration = False

            with self.lock:
                self.port_map = temp_map
            time.sleep(0.1)

    def _get_flow_key(self, si, di, sp, dp, pr):
        a, b = (si, sp), (di, dp)
        return (min(a, b), max(a, b), pr)

    def _handle_packet(self, packet):
        try:
            if not (packet.haslayer(IP) or packet.haslayer(IPv6)):
                return
            if not (packet.haslayer(TCP) or packet.haslayer(UDP)):
                return

            ip_layer = packet[IP] if packet.haslayer(IP) else packet[IPv6]
            layer    = packet[TCP] if packet.haslayer(TCP) else packet[UDP]
            proto    = "TCP" if packet.haslayer(TCP) else "UDP"

            src_ip, dst_ip = ip_layer.src, ip_layer.dst
            sport, dport   = layer.sport, layer.dport
            key = self._get_flow_key(src_ip, dst_ip, sport, dport, proto)
        except Exception as e:
            # Pakiety zniekształcone/fragmentowane czasem zgłaszają haslayer=True,
            # ale rzucają błąd przy faktycznym indeksowaniu warstwy — ignorujemy je.
            print(f"[!] Pominięto uszkodzony pakiet: {e}")
            return
        
        app = self.flow_cache.get(key)

        if not app:
            local_port = sport if src_ip in self.local_ips else dport
            with self.lock:
                app = self.port_map.get(local_port)
            if app:
                self.flow_cache[key] = app

        if app:
            self._save_raw_pcap(app, packet)
            self._debug_packet_count += 1
            # Wypisze KAŻDY pakiet Spotify i wymusi odświeżenie konsoli (flush=True)
            print(f"[*] Złapano {self._debug_packet_count} pakietów dla {app}!   ", end="\r", flush=True)
        else:
            # Wypisze kropkę dla KAŻDEGO innego pakietu w tle (żebyś wiedział, że sniffer w ogóle żyje)
            print(".", end="", flush=True)

    def _save_raw_pcap(self, app, packet):
        date_str = datetime.now().strftime("%Y-%m-%d")
        path = os.path.join(DATA_RAW_DIR, f"{app}_{date_str}.pcap")
        
        with self.lock:
            if app not in self.writers:
                self.writers[app] = PcapWriter(path, append=True, sync=True)
            self.writers[app].write(packet)

    def _detect_active_iface(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
        finally:
            s.close()

        for iface in get_if_list():
            try:
                if get_if_addr(iface) == local_ip:
                    print(f"[*] Wykryto aktywną kartę: {iface} ({local_ip})")
                    return iface
            except Exception:
                continue
        print("[!] Nie udało się wykryć aktywnej karty — Scapy użyje domyślnej (conf.iface).")
        return None

    def start(self):
        print(f"[*] Monitoring apps: {self.target_apps}")
        threading.Thread(target=self._port_mapping_loop, daemon=True).start()

        iface = self._detect_active_iface()

        print("[*] Starting capture. Press Ctrl+C to stop.")
        while not self.stop_event.is_set():
            try:
                sniff(
                    filter="ip or ip6",
                    prn=self._handle_packet,
                    store=False,
                    iface=iface,
                    stop_filter=lambda _: self.stop_event.is_set()
                )
            except KeyboardInterrupt:
                self.stop_event.set()
                print("\n[*] Stopping capture...")
                break
            except OSError as e:
                print(f"[!] Socket sniffer padł ({e}), restartuję capture...")
                time.sleep(1)
                continue

        with self.lock:
            for w in self.writers.values():
                w.close()
        print("[*] All files saved to", DATA_RAW_DIR)