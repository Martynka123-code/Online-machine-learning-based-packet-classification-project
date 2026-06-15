import json
import torch
import numpy as np
from models.cnn_trainer import OptimizedPacketCNN
from preprocessing.cnn_preprocessor import CNNPreprocessor
from scapy.layers.inet import TCP, UDP

class CNNOnline:
    def __init__(self, model_path, class_map_path):
        print(f"[*] Loading CNN model checkpoint from: {model_path}")
        try:
            # Load PyTorch Lightning checkpoint
            self.model = OptimizedPacketCNN.load_from_checkpoint(model_path)
            self.model.eval()  # Set model to evaluation/inference mode
            torch.set_grad_enabled(False)  # Disable gradient calculation for speed
            print("[+] CNN Model loaded successfully.")
        except Exception as e:
            print(f"[!] Error loading CNN model: {e}")
            self.model = None

        print(f"[*] Loading class mapping from: {class_map_path}")
        try:
            with open(class_map_path, 'r') as f:
                self.class_map = json.load(f)
            # Invert the dictionary to map index -> application name
            self.idx_to_app = {int(v): k for k, v in self.class_map.items()}
            print(f"[+] Loaded class mapping: {self.idx_to_app}")
        except Exception as e:
            print(f"[!] Error loading class map: {e}")
            self.idx_to_app = {}

        print("[*] Wykrywanie lokalnych adresów IP dla modelu CNN...")
        # Korzystamy z ekstraktora RF do wykrycia fizycznych IP tego komputera
        self.local_ips = FlowFeatureExtractor()._get_local_ips()
        
        # Przekazujemy adresy do preprocesora żeby odpowiednio flagował pakiety In/Out
        self.preprocessor = CNNPreprocessor(max_length=1000, local_ips=self.local_ips)

    def predict_packet(self, packet):
        if self.model is None:
            return None, 0.0

        if not (packet.haslayer(TCP) or packet.haslayer(UDP)):
            return None, 0.0

        # Dodaj odrzucanie szumu na żywo
        if is_noise_packet(packet):
            return None, 0.0

        try:
            byte_array = self.preprocessor.packet_to_bytes(packet)
            if byte_array is None: # Zabezpieczenie przed błędami parsowania L3
                return None, 0.0
            
            x = np.expand_dims(byte_array, axis=(0, 1))
            x_tensor = torch.tensor(x, dtype=torch.float32)

            logits = self.model(x_tensor)
            probabilities = torch.softmax(logits, dim=1)[0]
            
            max_prob, predicted_idx = torch.max(probabilities, dim=0)
            predicted_idx = int(predicted_idx.item())
            confidence = float(max_prob.item())

            predicted_app = self.idx_to_app.get(predicted_idx, "UNKNOWN")
            return predicted_app, confidence
        except Exception as e:
            return f"ERROR ({e})", 0.0
        
    def predict_batch(self, packets):
        """Preprocesses a list of packets and performs fast batched inference."""
        if self.model is None or not packets:
            return []

        valid_packets = []
        byte_arrays = []

        # 1. Błyskawiczny Preprocessing dla całej paczki
        for packet in packets:
            if not (packet.haslayer(TCP) or packet.haslayer(UDP)):
                continue
                
            # Załóż, że zaimportowałeś is_noise_packet w tym pliku:
            # if is_noise_packet(packet): continue 
                
            b_arr = self.preprocessor.packet_to_bytes(packet)
            if b_arr is not None:
                byte_arrays.append(b_arr)
                valid_packets.append(packet)

        if not byte_arrays:
            return []

        # 2. Sklejamy z procesora macierze do jednego tensora PyTorch
        # Zmienia kształt z 32x (1000,) na (32, 1, 1000)
        batch_x = np.array(byte_arrays)
        batch_x = np.expand_dims(batch_x, axis=1)
        x_tensor = torch.tensor(batch_x, dtype=torch.float32)

        # 3. Jedna SZYBKA inferencja dla wszystkich
        results = []
        try:
            with torch.no_grad():
                logits = self.model(x_tensor)
                probabilities = torch.softmax(logits, dim=1)
                max_probs, predicted_idxs = torch.max(probabilities, dim=1)
                
            for i in range(len(valid_packets)):
                idx = int(predicted_idxs[i].item())
                conf = float(max_probs[i].item())
                app = self.idx_to_app.get(idx, "UNKNOWN")
                results.append((valid_packets[i], app, conf))
                
            return results
        except Exception as e:
            print(f"[!] Batch Prediction Error: {e}")
            return []

