import torch
import torch.nn as nn
import numpy as np


class PhysicsInformedLoss(nn.Module):
    """
    Combined data + physics loss for heat diffusion surrogate.

    L_total = L_data + lambda_energy * L_energy
                     + lambda_bc     * L_bc
                     + lambda_pde    * L_pde

    Parameters
    ----------
    lambda_energy : float  Weight for energy conservation loss.
    lambda_bc     : float  Weight for boundary condition loss.
    lambda_pde    : float  Weight for PDE residual loss.
    alpha         : float  Thermal diffusivity (for PDE residual).
    dx            : float  Grid spacing (for PDE residual).
    dt            : float  Time step (for PDE residual).
    """

    def __init__(
        self,
        lambda_energy = 1.0,
        lambda_bc     = 1.0,
        lambda_pde    = 0.0,
        alpha         = 0.01,
        dx            = 1.0 / 63,
        dt            = None,
    ):
        super().__init__()
        self.lambda_energy = lambda_energy
        self.lambda_bc     = lambda_bc
        self.lambda_pde    = lambda_pde
        self.alpha         = alpha
        self.dx            = dx

        # compute stable dt if not provided
        if dt is None:
            self.dt = 0.9 * dx**2 / (4.0 * alpha)
        else:
            self.dt = dt

        self.r = alpha * self.dt / dx**2
        assert self.r <= 0.25, f"Unstable r={self.r:.4f}"

    def data_loss(self, pred, target):
        """Standard MSE against solver ground truth."""
        return nn.functional.mse_loss(pred, target)

    def energy_loss(self, pred, inputs):
        """
        Penalize predictions where mean temperature exceeds input mean.
        Uses ReLU so only violations are penalized, not correct behavior.

        Physics: for Dirichlet BC, total energy must decrease each step.
        """
        pred_mean  = pred.mean(dim=(-2, -1))    # mean per sample
        input_mean = inputs.mean(dim=(-2, -1))  # mean per sample

        # ReLU: only penalize when pred_mean > input_mean
        violation = torch.relu(pred_mean - input_mean)
        return violation.mean()

    def bc_loss(self, pred):
        """
        Penalize non-zero values at the four domain boundaries.
        Physics: Dirichlet BC requires T=0 at all edges.
        """
        # pred shape: (B, 1, H, W)
        top    = pred[:, :, 0,  :].pow(2).mean()
        bottom = pred[:, :, -1, :].pow(2).mean()
        left   = pred[:, :, :,  0].pow(2).mean()
        right  = pred[:, :, :, -1].pow(2).mean()
        return (top + bottom + left + right) / 4.0

    def pde_residual_loss(self, pred, inputs):
        """
        Penalize deviation from one explicit finite-difference step.

        Computes T_fd = one FD step applied to inputs,
        then measures MSE(pred, T_fd).

        This enforces single-step physical consistency
        independent of the stride used in data generation.

        Note: this is computed on interior points only.
        """
        T = inputs  # (B, 1, H, W)
        r = self.r

        # explicit FD update on interior points
        T_fd = T.clone()
        T_fd[:, :, 1:-1, 1:-1] = (
            T[:, :, 1:-1, 1:-1]
            + r * (
                T[:, :, 2:,   1:-1]
              + T[:, :, :-2,  1:-1]
              + T[:, :, 1:-1, 2:]
              + T[:, :, 1:-1, :-2]
              - 4 * T[:, :, 1:-1, 1:-1]
            )
        )
        # zero boundaries
        T_fd[:, :, 0,  :] = 0.0
        T_fd[:, :, -1, :] = 0.0
        T_fd[:, :, :,  0] = 0.0
        T_fd[:, :, :, -1] = 0.0

        # compare CNN prediction to one FD step
        # only on interior points
        return nn.functional.mse_loss(
            pred[:, :, 1:-1, 1:-1],
            T_fd[:, :, 1:-1, 1:-1]
        )

    def forward(self, pred, target, inputs):
        """
        Compute total physics-informed loss.

        Parameters
        ----------
        pred   : torch.Tensor (B, 1, H, W)  CNN prediction
        target : torch.Tensor (B, 1, H, W)  Solver ground truth
        inputs : torch.Tensor (B, 1, H, W)  Input field T(t)

        Returns
        -------
        total_loss : torch.Tensor  Scalar loss for backprop
        loss_dict  : dict          Individual loss components
        """
        l_data   = self.data_loss(pred, target)
        l_energy = self.energy_loss(pred, inputs)
        l_bc     = self.bc_loss(pred)
        l_pde    = self.pde_residual_loss(pred, inputs) \
                   if self.lambda_pde > 0 else torch.tensor(0.0)

        total = (
            l_data
            + self.lambda_energy * l_energy
            + self.lambda_bc     * l_bc
            + self.lambda_pde    * l_pde
        )

        return total, {
            "data"   : l_data.item(),
            "energy" : l_energy.item(),
            "bc"     : l_bc.item(),
            "pde"    : l_pde.item() if self.lambda_pde > 0 else 0.0,
            "total"  : total.item(),
        }