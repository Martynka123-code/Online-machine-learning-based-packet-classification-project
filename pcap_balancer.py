import os
from scapy.all import PcapReader, PcapWriter, RawPcapReader

def merge_and_split_pcaps(input_dir, output_dir, num_parts=3):
    """
    1. Znajduje pliki PCAP w input_dir.
    2. Grupuje je wg aplikacji (pierwszy człon nazwy pliku).
    3. Scala wszystkie pakiety danej aplikacji.
    4. Dzieli wynik na 'num_parts' równych plików.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Pobierz pliki i pogrupuj
    files = [f for f in os.listdir(input_dir) if f.endswith('.pcap')]
    class_files = {}
    for f in files:
        # np. 'discord_jakis_opis.pcap' -> 'discord'
        app_name = f.split('_')[0].lower()
        if app_name not in class_files:
            class_files[app_name] = []
        class_files[app_name].append(os.path.join(input_dir, f))

    # 2. Przetwarzaj każdą aplikację
    for app_name, file_paths in class_files.items():
        print(f"\n[*] Przetwarzanie klasy: {app_name.upper()}")
        
        # Szybkie liczenie pakietów
        total_packets = 0
        for fp in file_paths:
            try:
                total_packets += sum(1 for _ in RawPcapReader(fp))
            except Exception as e:
                print(f"    [!] Błąd czytania {fp}: {e}")

        if total_packets < num_parts:
            print(f"    [!] Za mało pakietów ({total_packets}), pomijam.")
            continue

        chunk_size = total_packets // num_parts
        print(f"    Suma pakietów: {total_packets}. Dzielę na {num_parts} plików po ~{chunk_size} pkt.")

        # 3. Scalanie i podział strumieniowy
        current_part = 1
        current_pkt_count = 0
        out_path = os.path.join(output_dir, f"{app_name}_balanced_part{current_part}.pcap")
        writer = PcapWriter(out_path, append=False, sync=False)

        for fp in file_paths:
            with PcapReader(fp) as reader:
                for pkt in reader:
                    writer.write(pkt)
                    current_pkt_count += 1

                    if current_pkt_count >= chunk_size and current_part < num_parts:
                        writer.close()
                        print(f"      -> Zapisano {os.path.basename(out_path)} ({current_pkt_count} pkt)")
                        current_part += 1
                        current_pkt_count = 0
                        out_path = os.path.join(output_dir, f"{app_name}_balanced_part{current_part}.pcap")
                        writer = PcapWriter(out_path, append=False, sync=False)

        writer.close()
        print(f"      -> Zapisano {os.path.basename(out_path)} ({current_pkt_count} pkt)")

if __name__ == "__main__":
    # Foldery
    INPUT_DIR = "data/raw_pcaps"
    OUTPUT_DIR = "data/balanced_pcaps"
    
    print(f"[*] Rozpoczynam łączenie i podział: {INPUT_DIR} -> {OUTPUT_DIR}")
    merge_and_split_pcaps(INPUT_DIR, OUTPUT_DIR, num_parts=3)
    print("\n[+] GOTOWE! Przenieś pliki z balanced_pcaps do raw_pcaps i usuń stare.")