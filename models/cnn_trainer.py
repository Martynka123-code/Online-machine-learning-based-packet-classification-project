import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from pytorch_lightning import LightningModule, Callback
from torchmetrics.functional import f1_score


class MetricsCallback(Callback):
    """Zbiera metryki per-epoka z trainer.callback_metrics."""

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
    """Wczytuje dataset CNN z pliku .npz."""

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
        # Dodajemy wymiar kanału: (1000,) → (1, 1000)
        x = np.expand_dims(x, axis=0)
        return {
            "feature": torch.tensor(x, dtype=torch.float32),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }


class OptimizedPacketCNN(LightningModule):
    """
    1D-CNN do klasyfikacji ruchu sieciowego na poziomie bajtów pakietu.

    Poprawki względem poprzedniej wersji:
    - padding obliczany automatycznie z kernel_size (same padding)
    - spójne wymiary warstw FC: max_pool_out → 128 → 64 → 32 → output_dim
    - val_f1_macro logowane w validation_step (potrzebne przez MetricsCallback)
    - dodany output_dim do save_hyperparameters (wymagany przy load_from_checkpoint)
    """

    def __init__(
        self,
        output_dim: int = 5,
        signal_length: int = 1000,
        learning_rate: float = 0.0007478604002657379,
        conv1_filters: int = 64,
        conv2_filters: int = 32,
        kernel_size: int = 4,
        dropout: float = 0.4895229008693801,
    ):
        super().__init__()
        # Zapisuje WSZYSTKIE parametry — konieczne dla load_from_checkpoint
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        # ── POPRAWKA 1: padding = kernel_size // 2  →  "same-ish" padding ──
        # Zapobiega NameError który powodował crash przy każdej próbie treningu.
        padding = kernel_size // 2

        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=conv1_filters,
                kernel_size=kernel_size,
                stride=2,
                padding=padding,
            ),
            nn.BatchNorm1d(conv1_filters),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=conv1_filters,
                out_channels=conv2_filters,
                kernel_size=kernel_size,
                stride=2,
                padding=padding,
            ),
            nn.BatchNorm1d(conv2_filters),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        # Automatyczne obliczenie rozmiaru wejścia do FC
        with torch.no_grad():
            dummy_x = torch.zeros(1, 1, signal_length)
            dummy_x = self.conv1(dummy_x)
            dummy_x = self.conv2(dummy_x)
            flatten_size = dummy_x.view(1, -1).shape[1]

        print(f"[CNN] Flatten size after conv layers: {flatten_size}")

        # ── POPRAWKA 2: spójne wymiary FC ──
        # Poprzednio: Linear(flatten_size, 128) → Linear(64, 32)  ← błąd wymiaru
        # Teraz:      Linear(flatten_size, 128) → Linear(128, 64) → Linear(64, 32)
        self.fc = nn.Sequential(
            nn.Linear(in_features=flatten_size, out_features=128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=128, out_features=64),   # ← poprawione 128→64
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
        )

        self.out = nn.Linear(in_features=32, out_features=self.hparams.output_dim)

    # ── Forward ────────────────────────────────────────────────────────────

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        x = self.out(x)
        return x

    # ── Training step ──────────────────────────────────────────────────────

    def training_step(self, batch, batch_idx):
        x = batch["feature"]
        y = batch["label"].long()
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    # ── Validation step ────────────────────────────────────────────────────

    def validation_step(self, batch, batch_idx):
        x = batch["feature"]
        y = batch["label"].long()
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()

        # ── POPRAWKA 3: logowanie val_f1_macro ──
        # Poprzednio brak tego loga → MetricsCallback zawsze miał pustą listę val_f1_macro
        # → wykres F1 nigdy się nie generował, wyniki nie były zbierane
        num_classes = self.hparams.output_dim
        f1 = f1_score(
            preds, y,
            task="multiclass",
            num_classes=num_classes,
            average="macro",
        )

        self.log("val_loss",     loss, prog_bar=True,  on_epoch=True, on_step=False)
        self.log("val_acc",      acc,  prog_bar=True,  on_epoch=True, on_step=False)
        self.log("val_f1_macro", f1,   prog_bar=True,  on_epoch=True, on_step=False)

        return loss

    # ── Optimizer ──────────────────────────────────────────────────────────

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4,   # drobna regularyzacja L2 — zmniejsza overfitting
        )
        # Scheduler: zmniejsza LR gdy val_loss przestaje spadać
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=3,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }