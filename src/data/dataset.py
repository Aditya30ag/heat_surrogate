import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset


class HeatDiffusionDataset(Dataset):
    """
    PyTorch Dataset for single-step heat diffusion prediction.

    Each item is:
        input  : torch.Tensor, shape (1, N, N)  — temperature field at t
        target : torch.Tensor, shape (1, N, N)  — temperature field at t+stride*dt

    The channel dimension (1) is added here so the CNN receives
    properly shaped input without any extra transforms.

    Parameters
    ----------
    inputs  : np.ndarray, shape (n, N, N)
    targets : np.ndarray, shape (n, N, N)
    """

    def __init__(self, inputs, targets):
        assert inputs.shape == targets.shape, \
            f"Shape mismatch: inputs {inputs.shape} vs targets {targets.shape}"

        # store as float32 tensors with channel dimension added
        self.inputs  = torch.from_numpy(inputs).unsqueeze(1).float()   # (n, 1, N, N)
        self.targets = torch.from_numpy(targets).unsqueeze(1).float()  # (n, 1, N, N)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def load_dataset(path="../../data/heat_dataset.npz"):
    """
    Load the saved dataset and return PyTorch Dataset objects
    for train, validation, and test splits.

    Parameters
    ----------
    path : str  Path to the .npz file.

    Returns
    -------
    train_ds, val_ds, test_ds : HeatDiffusionDataset
    metadata : dict
    """
    path = Path(path)
    assert path.exists(), f"Dataset file not found: {path}"

    data = np.load(path)

    train_ds = HeatDiffusionDataset(
        data["train_inputs"],
        data["train_targets"]
    )
    val_ds = HeatDiffusionDataset(
        data["val_inputs"],
        data["val_targets"]
    )
    test_ds = HeatDiffusionDataset(
        data["test_inputs"],
        data["test_targets"]
    )

    # reconstruct metadata from saved arrays
    metadata = {
        k.replace("meta_", ""): data[k].item()
        for k in data.files
        if k.startswith("meta_")
    }

    print("Dataset loaded successfully.")
    print(f"  Train : {len(train_ds)} samples")
    print(f"  Val   : {len(val_ds)} samples")
    print(f"  Test  : {len(test_ds)} samples")
    print(f"  Input shape per sample : {train_ds[0][0].shape}")
    print(f"  Metadata: {metadata}")

    return train_ds, val_ds, test_ds, metadata