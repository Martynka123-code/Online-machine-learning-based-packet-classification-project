# models/cnn_trainer.py
import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from pytorch_lightning import LightningModule

class PacketByteDataset(Dataset):
    def __init__(self, npz_path):
        print(f"[*] Loading CNN dataset into memory from: {npz_path}")
        data = np.load(npz_path)
        self.features = data['features']
        self.labels = data['labels']
        print(f"[+] Loaded {len(self.features)} packets.")
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        # 1D CNN expects shape: [channels, length] -> [1, 1000]
        x = self.features[idx]
        x = np.expand_dims(x, axis=0) 
        
        return {
            "feature": torch.tensor(x, dtype=torch.float32),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }

class OptimizedPacketCNN(LightningModule):
    # Udostępniamy parametry w konstruktorze z bezpiecznymi domyślnymi wartościami
    def __init__(self, output_dim=5, signal_length=1000, learning_rate=0.001,
                 conv1_filters=80, conv2_filters=40, kernel_size=3, dropout=0.5):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        # Pierwszy blok konwolucyjny
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=conv1_filters,
                kernel_size=kernel_size,
                stride=2,
                padding=kernel_size // 2  # Dynamiczny padding chroniący przed błędem wymiaru
            ),
            nn.BatchNorm1d(conv1_filters),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        # Drugi blok konwolucyjny
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=conv1_filters,
                out_channels=conv2_filters,
                kernel_size=kernel_size,
                stride=2,
                padding=kernel_size // 2
            ),
            nn.BatchNorm1d(conv2_filters),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        # Trik z dummy_x automatycznie dostosuje wielkość warstwy liniowej
        dummy_x = torch.rand(1, 1, self.hparams.signal_length, requires_grad=False)
        dummy_x = self.conv1(dummy_x)
        dummy_x = self.conv2(dummy_x)
        max_pool_out = dummy_x.view(1, -1).shape[1]

        # Warstwy w pełni połączone (Klasyfikator gęsty z poprawnym BatchNorm)
        self.fc = nn.Sequential(
            nn.Linear(in_features=max_pool_out, out_features=25),
            nn.BatchNorm1d(25),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=25, out_features=10),
            nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

        # Warstwa wyjściowa
        self.out = nn.Linear(in_features=10, out_features=self.hparams.output_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        x = self.out(x)
        return x

    def training_step(self, batch, batch_idx):
        x = batch["feature"]
        y = batch["label"].long()
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["feature"]
        y = batch["label"].long()
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()

        # Obliczanie F1-Macro (Twoja implementacja)
        from sklearn.metrics import f1_score
        f1 = f1_score(
            y.cpu().numpy(),
            preds.cpu().numpy(),
            average="macro",
            zero_division=0
        )

        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_epoch=True)
        self.log("val_f1_macro", torch.tensor(f1), prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
