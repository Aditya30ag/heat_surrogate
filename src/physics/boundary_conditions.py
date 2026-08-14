import numpy as np


def apply_dirichlet_zero(T):
    """
    Apply homogeneous Dirichlet boundary conditions.
    Sets all four edges of the 2D temperature field to zero.

    Parameters
    ----------
    T : np.ndarray, shape (N, N)
        Temperature field (modified in place).

    Returns
    -------
    T : np.ndarray
        Temperature field with edges set to zero.
    """
    T[0, :]  = 0.0   # top edge
    T[-1, :] = 0.0   # bottom edge
    T[:, 0]  = 0.0   # left edge
    T[:, -1] = 0.0   # right edge
    return T


def apply_neumann_zero(T):
    """
    Apply homogeneous Neumann boundary conditions (zero flux).
    Copies the adjacent interior value to each boundary cell,
    which enforces dT/dn = 0 at every edge.

    Parameters
    ----------
    T : np.ndarray, shape (N, N)
        Temperature field (modified in place).

    Returns
    -------
    T : np.ndarray
        Temperature field with zero-flux edges.
    """
    T[0, :]  = T[1, :]    # top edge mirrors row below it
    T[-1, :] = T[-2, :]   # bottom edge mirrors row above it
    T[:, 0]  = T[:, 1]    # left edge mirrors column to its right
    T[:, -1] = T[:, -2]   # right edge mirrors column to its left
    return T