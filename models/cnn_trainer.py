# models/cnn_trainer.py
import torch
from torch import nn as nn
from torch.nn import functional as F
from pytorch_lightning import LightningModule


class OptimizedPacketCNN(LightningModule):
    def __init__(self, output_dim=6, signal_length=1000, learning_rate=0.001):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate

        # Pierwszy blok konwolucyjny (z zapożyczoną konfiguracją 512 filtrów, stride=2)
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=512,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        # Drugi blok konwolucyjny (256 filtrów, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=512,
                out_channels=256,
                kernel_size=3,
                stride=2,
                padding=1
            ),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        # Automatyczne wyliczenie rozmiaru wejściowego dla warstw gęstych (Trik z dummy_x)
        dummy_x = torch.rand(1, 1, self.hparams.signal_length, requires_grad=False)
        dummy_x = self.conv1(dummy_x)
        dummy_x = self.conv2(dummy_x)
        max_pool_out = dummy_x.view(1, -1).shape[1]

        # Warstwy w pełni połączone (Klasyfikator gęsty z dropoutem 0.5 )
        self.fc = nn.Sequential(
            nn.Linear(in_features=max_pool_out, out_features=128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=128, out_features=32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )

        # Warstwa wyjściowa (Logits)
        self.out = nn.Linear(in_features=32, out_features=self.hparams.output_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        x = self.out(x)
        return x

    def training_step(self, batch, batch_idx):
        x = batch["feature"].float()
        y = batch["label"].long()
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["feature"].float()
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
