import os
import pickle
import torch
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from models.cnn_trainer import OptimizedPacketCNN, PacketByteDataset

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "data", "cnn_datasets", "cnn_dataset_master.npz")

# Search space definition
space = {
    'learning_rate': hp.loguniform('learning_rate', torch.log(torch.tensor(1e-4)).item(),
                                   torch.log(torch.tensor(1e-2)).item()),
    'dropout': hp.uniform('dropout', 0.1, 0.6),
    'conv_filters': hp.choice('conv_filters', [
        {'c1': 32, 'c2': 16},
        {'c1': 48, 'c2': 24},
        {'c1': 64, 'c2': 32},
        {'c1': 80, 'c2': 40},
        {'c1': 100, 'c2': 50}
    ]),
    'kernel_size': hp.choice('kernel_size', [2, 3, 5, 7, 9]),
    'batch_size': hp.choice('batch_size', [16, 32, 64, 128, 256, 512])
}

# Load the full dataset into memory before starting the Hyperopt loop
full_dataset = PacketByteDataset(DATASET_PATH)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])


def objective(params):
    print(f"\n==================================================")
    print(f"[>] Configuration testing:\n{params}")
    print(f"==================================================")

    # Create dynamic DataLoaders using the batch_size parameter from Hyperopt
    train_loader = DataLoader(train_dataset, batch_size = params['batch_size'], shuffle = True, num_workers = 0)
    val_loader = DataLoader(val_dataset, batch_size = params['batch_size'], shuffle = False, num_workers = 0)

    model = OptimizedPacketCNN(
        learning_rate = params['learning_rate'],
        dropout = params['dropout'],
        conv1_filters = params['conv_filters']['c1'],
        conv2_filters = params['conv_filters']['c2'],
        kernel_size = params['kernel_size']
    )

    # Early Stopping - stops the trial if the model fails to improve after 2 epochs
    early_stop = EarlyStopping(monitor="val_loss", mode="min", patience=2)

    trainer = Trainer(
        max_epochs = 4,
        accelerator = "auto",
        callbacks = [early_stop],
        enable_progress_bar = True,
        logger = False
    )

    try:
        # Model training
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        # Extract the best (lowest) validation loss value
        val_loss = trainer.callback_metrics.get("val_loss").item()

        return {'loss': val_loss, 'status': STATUS_OK}
    except Exception as e:
        print(f"[!] Trial failed due to an architecture error: {e}")
        return {'status': 'fail', 'failure_reason': str(e)}


if __name__ == "__main__":
    trials = Trials()

    print("[*] Starting TPE Bayesian Optimization (10 iterations)...")
    best = fmin(
        fn = objective,
        space = space,
        algo = tpe.suggest,
        max_evals = 8,
        trials = trials
    )

    print("\n=============================================")
    print("FINISHED! BEST HYPERPARAMETER INDEXES:")
    print(best)
    print("=============================================")

    # Save the full experiment history to a binary file
    with open("hyperopt_trials.pkl", "wb") as f:
        pickle.dump(trials, f)
    print("[+] Successfully saved full experiment history to file: hyperopt_trials.pkl")