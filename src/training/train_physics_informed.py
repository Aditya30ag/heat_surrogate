import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import json
import time
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "models"))
sys.path.append(str(Path(__file__).resolve().parents[1] / "data"))

from cnn import HeatSurrogateCNN
from physics_informed import PhysicsInformedLoss
from dataset import load_dataset


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train_physics_informed(
    dataset_path   = "../../data/heat_dataset.npz",
    save_dir       = "../../results",
    lambda_energy  = 1.0,
    lambda_bc      = 1.0,
    lambda_pde     = 0.0,
    n_filters      = 32,
    batch_size     = 32,
    n_epochs       = 100,
    lr             = 1e-3,
    patience       = 15,
    seed           = 42,
    run_name       = "pi_cnn",
):
    """
    Train a physics-informed CNN surrogate.

    Same architecture as baseline CNN but with physics loss terms
    added to the standard MSE data loss.
    """
    set_seed(seed)

    save_dir = Path(save_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device      : {device}")
    print(f"Run name    : {run_name}")
    print(f"λ_energy    : {lambda_energy}")
    print(f"λ_bc        : {lambda_bc}")
    print(f"λ_pde       : {lambda_pde}")

    # load dataset and metadata
    train_ds, val_ds, _, metadata = load_dataset(dataset_path)

    dx = float(metadata["dx"])
    dt = float(metadata["dt"])

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                               shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size*2,
                               shuffle=False, num_workers=0)

    model = HeatSurrogateCNN(n_filters=n_filters).to(device)
    print(f"Parameters  : {model.count_parameters():,}")

    # physics-informed loss
    loss_fn = PhysicsInformedLoss(
        lambda_energy = lambda_energy,
        lambda_bc     = lambda_bc,
        lambda_pde    = lambda_pde,
        alpha         = float(metadata["alpha"]),
        dx            = dx,
        dt            = dt,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=7
    )

    # tracking
    history = {k: [] for k in
               ["train_total", "train_data", "train_energy",
                "train_bc", "val_data", "val_total"]}

    best_val_data_loss = float("inf")
    epochs_no_improve  = 0
    best_epoch         = 0
    t_start            = time.time()

    print(f"\n{'Epoch':>6} {'TrainTotal':>12} {'TrainData':>12} "
          f"{'Energy':>10} {'BC':>10} {'ValData':>10}")
    print("-" * 65)

    for epoch in range(1, n_epochs + 1):

        # ── training ─────────────────────────────────
        model.train()
        epoch_losses = {k: 0.0 for k in
                        ["total", "data", "energy", "bc"]}

        for inputs, targets in train_loader:
            inputs  = inputs.to(device)
            targets = targets.to(device)

            predictions = model(inputs)

            # physics-informed loss needs inputs for energy/BC/PDE terms
            total_loss, loss_dict = loss_fn(predictions, targets, inputs)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            bs = inputs.size(0)
            for k in epoch_losses:
                epoch_losses[k] += loss_dict[k] * bs

        n_train = len(train_loader.dataset)
        for k in epoch_losses:
            epoch_losses[k] /= n_train

        # ── validation ───────────────────────────────
        model.eval()
        val_data_loss  = 0.0
        val_total_loss = 0.0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs  = inputs.to(device)
                targets = targets.to(device)
                pred    = model(inputs)

                # validation: report DATA loss only (fair comparison
                # against baseline which used pure MSE validation)
                vl_data = nn.functional.mse_loss(pred, targets)
                vl_tot, _ = loss_fn(pred, targets, inputs)

                val_data_loss  += vl_data.item() * inputs.size(0)
                val_total_loss += vl_tot.item()  * inputs.size(0)

        val_data_loss  /= len(val_loader.dataset)
        val_total_loss /= len(val_loader.dataset)

        # schedule on data loss — not total loss
        # (avoids reducing lr just because physics term is large)
        scheduler.step(val_data_loss)

        # record history
        history["train_total"].append(epoch_losses["total"])
        history["train_data"].append(epoch_losses["data"])
        history["train_energy"].append(epoch_losses["energy"])
        history["train_bc"].append(epoch_losses["bc"])
        history["val_data"].append(val_data_loss)
        history["val_total"].append(val_total_loss)

        print(f"{epoch:>6} {epoch_losses['total']:>12.6f} "
              f"{epoch_losses['data']:>12.6f} "
              f"{epoch_losses['energy']:>10.6f} "
              f"{epoch_losses['bc']:>10.6f} "
              f"{val_data_loss:>10.6f}", flush=True)

        # checkpoint on validation DATA loss
        if val_data_loss < best_val_data_loss:
            best_val_data_loss = val_data_loss
            best_epoch         = epoch
            epochs_no_improve  = 0
            torch.save({
                "epoch"       : epoch,
                "model_state" : model.state_dict(),
                "val_data_loss": val_data_loss,
                "config": {
                    "lambda_energy": lambda_energy,
                    "lambda_bc"    : lambda_bc,
                    "lambda_pde"   : lambda_pde,
                    "n_filters"    : n_filters,
                },
            }, save_dir / "best_model.pt")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch}. "
                  f"Best val data loss: {best_val_data_loss:.6f} "
                  f"at epoch {best_epoch}.")
            break

    train_time = time.time() - t_start
    print(f"\nTraining complete in {train_time:.1f}s")

    history["best_epoch"]      = best_epoch
    history["best_val_data_loss"] = best_val_data_loss
    history["train_time"]      = train_time
    history["config"] = {
        "lambda_energy": lambda_energy,
        "lambda_bc"    : lambda_bc,
        "lambda_pde"   : lambda_pde,
    }

    with open(save_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    return model, history