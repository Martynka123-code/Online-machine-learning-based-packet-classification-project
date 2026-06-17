import os
import glob
import pickle
import torch
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from cnn_trainer import OptimizedPacketCNN, PacketByteDataset

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# POPRAWKA: szukaj najnowszego pliku .npz zamiast hardkodowanej ścieżki
_npz_files = sorted(glob.glob(os.path.join(BASE_DIR, "data", "cnn_datasets", "*.npz")))
if not _npz_files:
    raise FileNotFoundError(
        "Brak plików .npz w data/cnn_datasets/. "
        "Najpierw uruchom opcję 3 (Extract Features CNN) z main.py."
    )
DATASET_PATH = _npz_files[-1]
print(f"[*] optimize.py użyje datasetu: {DATASET_PATH}")

space = {
    'learning_rate': hp.loguniform('learning_rate',
                                   torch.log(torch.tensor(1e-4)).item(),
                                   torch.log(torch.tensor(1e-2)).item()),
    'dropout': hp.uniform('dropout', 0.1, 0.6),
    'conv_filters': hp.choice('conv_filters', [
        {'c1': 32,  'c2': 16},
        {'c1': 48,  'c2': 24},
        {'c1': 64,  'c2': 32},
        {'c1': 80,  'c2': 40},
        {'c1': 100, 'c2': 50}
    ]),
    'kernel_size': hp.choice('kernel_size', [2, 3, 5, 7, 9]),
    'batch_size':  hp.choice('batch_size',  [16, 32, 64, 128, 256, 512])
}

full_dataset = PacketByteDataset(DATASET_PATH)

num_classes = len(set(full_dataset.labels))
print(f"[*] Wykryto klas (output_dim): {num_classes}")

train_size = int(0.8 * len(full_dataset))
val_size   = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])


def objective(params):
    print(f"\n{'='*50}")
    print(f"[>] Configuration testing:\n{params}")
    print(f"{'='*50}")

    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True,  num_workers=0, drop_last=True)
    val_loader   = DataLoader(val_dataset,   batch_size=params['batch_size'], shuffle=False, num_workers=0)

    model = OptimizedPacketCNN(
        output_dim=num_classes,
        learning_rate=params['learning_rate'],
        dropout=params['dropout'],
        conv1_filters=params['conv_filters']['c1'],
        conv2_filters=params['conv_filters']['c2'],
        kernel_size=params['kernel_size']
    )

    early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=2)

    trainer = Trainer(
        max_epochs=4,
        accelerator="auto",
        callbacks=[early_stop],
        enable_progress_bar=True,
        logger=False
    )

    try:
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        val_loss = trainer.callback_metrics.get("val_loss").item()
        return {'loss': val_loss, 'status': STATUS_OK}
    except Exception as e:
        print(f"[!] Trial failed: {e}")
        return {'status': 'fail', 'failure_reason': str(e)}


if __name__ == "__main__":
    trials = Trials()
    print("[*] Starting TPE Bayesian Optimization (8 iterations)...")
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=20,
        trials=trials
    )

    print("\n" + "=" * 50)
    print("FINISHED! BEST HYPERPARAMETER INDEXES:")
    print(best)
    print("=" * 50)

    with open("hyperopt_trials.pkl", "wb") as f:
        pickle.dump(trials, f)
    print("[+] Saved full experiment history to: hyperopt_trials.pkl")