import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    A single building block: Conv2d → BatchNorm → ReLU.

    This pattern is standard in modern CNNs. BatchNorm stabilizes
    training by normalizing activations within each batch.
    ReLU introduces non-linearity — without it, stacking linear
    convolutions would still be a single linear operation.

    Parameters
    ----------
    in_channels  : int  Number of input feature maps.
    out_channels : int  Number of output feature maps (filters).
    kernel_size  : int  Filter size (default 3).
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2   # keeps spatial size unchanged

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size,
                      padding=padding,
                      bias=False),       # bias=False because BatchNorm has its own bias
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class HeatSurrogateCNN(nn.Module):
    """
    Plain CNN surrogate for single-step 2D heat diffusion prediction.

    Architecture:
        Input  (1, 64, 64)
        → ConvBlock(1  → 32)
        → ConvBlock(32 → 64)
        → ConvBlock(64 → 64)
        → ConvBlock(64 → 32)
        → Conv2d(32 → 1)       final layer, no activation
        Output (1, 64, 64)

    The final layer has NO activation function because temperature
    is a continuous unbounded value — we don't want to artificially
    clip predictions to [0,1]. The network should learn to predict
    the right range from data.

    Parameters
    ----------
    n_filters : int  Base number of filters (default 32).
                     Doubling this approximately 4x the parameter count.
    """

    def __init__(self, n_filters=32):
        super().__init__()

        self.encoder = nn.Sequential(
            ConvBlock(1,            n_filters),       # (1,64,64)  → (32,64,64)
            ConvBlock(n_filters,    n_filters * 2),   # (32,64,64) → (64,64,64)
            ConvBlock(n_filters*2,  n_filters * 2),   # (64,64,64) → (64,64,64)
            ConvBlock(n_filters*2,  n_filters),       # (64,64,64) → (32,64,64)
        )

        # final projection back to 1 channel — the predicted temperature field
        self.output_layer = nn.Conv2d(
            n_filters, 1,
            kernel_size=1,    # 1x1 conv — purely a channel mixer, no spatial mixing
            bias=True
        )

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor, shape (batch, 1, H, W)

        Returns
        -------
        torch.Tensor, shape (batch, 1, H, W)
        """
        features = self.encoder(x)
        out      = self.output_layer(features)
        return out

    def count_parameters(self):
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)