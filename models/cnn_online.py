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

        self.preprocessor = CNNPreprocessor(max_length=1000)

    def predict_packet(self, packet):
        """Preprocesses a single packet and returns the predicted class name and confidence."""
        if self.model is None:
            return None, 0.0

        # Filter out non-TCP/UDP packets just like during training
        if not (packet.haslayer(TCP) or packet.haslayer(UDP)):
            return None, 0.0

        try:
            # 1. Convert live packet to normalized byte vector (1000,)
            byte_array = self.preprocessor.packet_to_bytes(packet)
            
            # 2. Reshape for 1D CNN: [batch_size, channels, length] -> [1, 1, 1000]
            x = np.expand_dims(byte_array, axis=(0, 1))
            x_tensor = torch.tensor(x, dtype=torch.float32)

            # 3. Model Inference
            logits = self.model(x_tensor)
            probabilities = torch.softmax(logits, dim=1)[0]
            
            # 4. Extract highest probability and its class index
            max_prob, predicted_idx = torch.max(probabilities, dim=0)
            predicted_idx = int(predicted_idx.item())
            confidence = float(max_prob.item())

            predicted_app = self.idx_to_app.get(predicted_idx, "UNKNOWN")
            return predicted_app, confidence
        except Exception as e:
            return f"ERROR ({e})", 0.0