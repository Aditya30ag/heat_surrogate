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
from dataset import load_dataset


def set_seed(seed=42):
    """Fix all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # makes CUDA operations deterministic (slightly slower)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    """
    Run one full pass through the training set.

    Returns
    -------
    avg_loss : float  Mean loss over all batches in this epoch.
    """
    model.train()   # activates BatchNorm and Dropout (if any) in training mode
    total_loss = 0.0

    for inputs, targets in loader:
        inputs  = inputs.to(device)
        targets = targets.to(device)

        # forward pass
        predictions = model(inputs)

        # compute loss
        loss = loss_fn(predictions, targets)

        # backward pass
        optimizer.zero_grad()   # clear gradients from previous batch
        loss.backward()         # compute gradients via backprop
        optimizer.step()        # update parameters

        total_loss += loss.item() * inputs.size(0)   # accumulate total (not mean)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    """
    Evaluate the model on a data loader without computing gradients.

    @torch.no_grad() disables gradient tracking — faster and uses less memory.

    Returns
    -------
    avg_loss : float  Mean loss over all batches.
    """
    model.eval()    # deactivates BatchNorm training behavior
    total_loss = 0.0

    for inputs, targets in loader:
        inputs  = inputs.to(device)
        targets = targets.to(device)

        predictions = model(inputs)
        loss        = loss_fn(predictions, targets)
        total_loss += loss.item() * inputs.size(0)

    return total_loss / len(loader.dataset)


def train(
    dataset_path  = "../data/heat_dataset.npz",
    save_dir      = "../results",
    n_filters     = 32,
    batch_size    = 32,
    n_epochs      = 100,
    lr            = 1e-3,
    seed          = 42,
    patience      = 15,     # early stopping: stop if val loss doesn't improve
):
    """
    Full training pipeline for the heat surrogate CNN.

    Parameters
    ----------
    dataset_path : str   Path to the .npz dataset file.
    save_dir     : str   Directory to save model checkpoints and plots.
    n_filters    : int   Base filter count for the CNN.
    batch_size   : int   Training batch size.
    n_epochs     : int   Maximum number of training epochs.
    lr           : float Initial learning rate.
    seed         : int   Random seed.
    patience     : int   Early stopping patience (epochs without improvement).
    """
    set_seed(seed)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── device ───────────────────────────────────────
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        device = torch.device("cuda:0")
    else:
        n_gpus = 0
        device = torch.device("cpu")

    print(f"Using device: {device}")
    print(f"Number of GPUs: {n_gpus}")

    if n_gpus > 0:
        for i in range(n_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # ── data ─────────────────────────────────────────
    train_ds, val_ds, test_ds, metadata = load_dataset(dataset_path)

    train_loader = DataLoader(
        train_ds,
        batch_size  = batch_size,
        shuffle     = True,        # shuffle every epoch for better generalization
        num_workers = 0,           # 0 = load on main process (safest for debugging)
        pin_memory  = device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = batch_size * 2,  # can use larger batch for evaluation (no gradients)
        shuffle     = False,
        num_workers = 0,
    )

    # ── model ─────────────────────────────────────────
    model = HeatSurrogateCNN(n_filters=n_filters).to(device)
    print(f"\nModel architecture:")
    print(model)
    print(f"\nTotal trainable parameters: {model.count_parameters():,}")

    # ── loss, optimizer, scheduler ───────────────────
    loss_fn   = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # reduce lr by factor 0.5 if val loss doesn't improve for 7 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode     = "min",
        factor   = 0.5,
        patience = 7,
        verbose  = True,
    )

    # ── training loop ─────────────────────────────────
    train_losses = []
    val_losses   = []
    lr_history   = []
    best_val_loss   = float("inf")
    epochs_no_improve = 0
    best_epoch       = 0

    print(f"\nStarting training for up to {n_epochs} epochs...")
    print(f"{'Epoch':>6}  {'Train Loss':>12}  {'Val Loss':>12}  {'LR':>12}")
    print("-" * 50)

    t_start = time.time()

    for epoch in range(1, n_epochs + 1):

        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss   = evaluate(model, val_loader, loss_fn, device)

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        lr_history.append(current_lr)

        print(f"{epoch:>6}  {train_loss:>12.6f}  {val_loss:>12.6f}  {current_lr:>12.2e}", flush=True)

        # ── checkpoint: save best model ───────────────
        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_epoch       = epoch
            epochs_no_improve = 0
            torch.save({
                "epoch"      : epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss"   : val_loss,
                "train_loss" : train_loss,
                "config"     : {
                    "n_filters"  : n_filters,
                    "batch_size" : batch_size,
                    "lr"         : lr,
                    "seed"       : seed,
                },
            }, save_dir / "best_model.pt")
        else:
            epochs_no_improve += 1

        # ── early stopping ────────────────────────────
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch}. "
                  f"Best val loss: {best_val_loss:.6f} at epoch {best_epoch}.")
            break

    t_elapsed = time.time() - t_start
    print(f"\nTraining complete in {t_elapsed:.1f}s")
    print(f"Best validation loss : {best_val_loss:.6f} (epoch {best_epoch})")

    # ── plot training curves ──────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(train_losses, label="Train MSE", color="steelblue")
    axes[0].plot(val_losses,   label="Val MSE",   color="tomato")
    axes[0].axvline(best_epoch - 1, color="gray", linestyle="--",
                    label=f"Best epoch ({best_epoch})")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].set_yscale("log")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(lr_history, color="mediumseagreen")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Learning Rate")
    axes[1].set_title("Learning Rate Schedule")
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / "training_curves.png", dpi=150)
    plt.show()
    print(f"Training curves saved to {save_dir}/training_curves.png")

    # ── save training history ─────────────────────────
    history = {
        "train_losses" : train_losses,
        "val_losses"   : val_losses,
        "lr_history"   : lr_history,
        "best_epoch"   : best_epoch,
        "best_val_loss": best_val_loss,
        "n_epochs_run" : len(train_losses),
        "training_time_seconds": t_elapsed,
    }
    with open(save_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"Training history saved to {save_dir}/training_history.json")
    return model, history


if __name__ == "__main__":
    model, history = train(
        dataset_path = "../../data/heat_dataset.npz",
        save_dir     = "../../results",
        n_filters    = 32,
        batch_size   = 32,
        n_epochs     = 100,
        lr           = 1e-3,
        seed         = 42,
        patience     = 15,
    )