import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from pytorch_lightning import LightningModule, Callback


class MetricsCallback(Callback):
    def __init__(self):
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_acc": [],
            "val_f1_macro": [],
        }

    def on_train_epoch_end(self, trainer, pl_module):
        logged = trainer.callback_metrics
        if "train_loss" in logged:
            self.history["train_loss"].append(float(logged["train_loss"]))

    def on_validation_epoch_end(self, trainer, pl_module):
        logged = trainer.callback_metrics
        if "val_loss" in logged:
            self.history["val_loss"].append(float(logged["val_loss"]))
        if "val_acc" in logged:
            self.history["val_acc"].append(float(logged["val_acc"]))
        if "val_f1_macro" in logged:
            self.history["val_f1_macro"].append(float(logged["val_f1_macro"]))


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
        x = self.features[idx]
        x = np.expand_dims(x, axis=0)
        return {
            "feature": torch.tensor(x, dtype=torch.float32),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }


class OptimizedPacketCNN(LightningModule):
    def __init__(self, output_dim=5, signal_length=1000, learning_rate=0.0049,
                 conv1_filters=80, conv2_filters=40, kernel_size=2, dropout=0.17):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        # Automatyczne obliczenie rozmiaru wejścia do warstw FC
        with torch.no_grad():
            dummy_x = torch.zeros(1, 1, signal_length)
            dummy_x = self.conv1(dummy_x)
            dummy_x = self.conv2(dummy_x)
            max_pool_out = dummy_x.view(1, -1).shape[1]

        # POPRAWKA: spójne wymiary — 25 → 25, nie 25 → 128
        self.fc = nn.Sequential(
            nn.Linear(in_features=max_pool_out, out_features=128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=128, out_features=32),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

        self.out = nn.Linear(in_features=32, out_features=self.hparams.output_dim)

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
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_acc", acc, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)