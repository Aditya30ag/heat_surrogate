import numpy as np
from boundary_conditions import apply_dirichlet_zero


def make_grid(N, L=1.0):
    """
    Create a uniform spatial grid on [0, L] x [0, L].

    Parameters
    ----------
    N : int
        Number of grid points along each axis.
    L : float
        Physical size of the domain (default 1.0).

    Returns
    -------
    x : np.ndarray, shape (N,)
    y : np.ndarray, shape (N,)
    dx : float
        Grid spacing.
    """
    x  = np.linspace(0, L, N)
    y  = np.linspace(0, L, N)
    dx = L / (N - 1)
    return x, y, dx


def compute_stable_dt(dx, alpha, safety=0.9):
    """
    Compute the maximum stable time step for the explicit
    finite-difference scheme in 2D.

    Stability condition: r = alpha * dt / dx^2 <= 1/4
    => dt_max = dx^2 / (4 * alpha)

    A safety factor < 1.0 keeps you comfortably below the limit.

    Parameters
    ----------
    dx    : float  Grid spacing.
    alpha : float  Thermal diffusivity.
    safety: float  Safety factor (default 0.9, so r = 0.225).

    Returns
    -------
    dt : float   Safe time step.
    r  : float   Fourier number (should be <= 0.25).
    """
    dt = safety * dx**2 / (4.0 * alpha)
    r  = alpha * dt / dx**2
    return dt, r


def gaussian_hotspot(x, y, cx=0.5, cy=0.5, sigma=0.1, amplitude=1.0):
    """
    Create a 2D Gaussian temperature distribution (a single hot blob).

    T0(x, y) = amplitude * exp(-((x-cx)^2 + (y-cy)^2) / (2*sigma^2))

    Parameters
    ----------
    x, y      : np.ndarray, shape (N,)  Grid coordinate vectors.
    cx, cy    : float  Center of the blob.
    sigma     : float  Width of the blob.
    amplitude : float  Peak temperature.

    Returns
    -------
    T0 : np.ndarray, shape (N, N)  Initial temperature field.
    """
    # meshgrid turns 1D coordinate vectors into 2D coordinate arrays
    X, Y = np.meshgrid(x, y)
    T0   = amplitude * np.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sigma**2))
    return T0


def step(T, r):
    """
    Advance the temperature field by one time step using the
    explicit finite-difference scheme.

    Update rule (interior points only):
    T_new[i,j] = T[i,j] + r * (T[i+1,j] + T[i-1,j]
                                + T[i,j+1] + T[i,j-1]
                                - 4*T[i,j])

    Boundary conditions (Dirichlet zero) are enforced after the update.

    Parameters
    ----------
    T : np.ndarray, shape (N, N)  Current temperature field.
    r : float                     Fourier number (alpha*dt/dx^2).

    Returns
    -------
    T_new : np.ndarray, shape (N, N)  Temperature field at next time step.
    """
    # allocate new array — never update T in place during the step,
    # because you need the OLD values of all neighbors simultaneously
    T_new = T.copy()

    # vectorized update of all interior points in one line
    T_new[1:-1, 1:-1] = (
        T[1:-1, 1:-1]
        + r * (
            T[2:,   1:-1]   # T[i+1, j]
          + T[:-2,  1:-1]   # T[i-1, j]
          + T[1:-1, 2:]     # T[i, j+1]
          + T[1:-1, :-2]    # T[i, j-1]
          - 4 * T[1:-1, 1:-1]
        )
    )

    # re-apply boundary conditions (Dirichlet zero)
    T_new = apply_dirichlet_zero(T_new)
    return T_new


def run_simulation(T0, alpha, dx, n_steps, save_every=1):
    """
    Run the full heat diffusion simulation from initial condition T0.

    Parameters
    ----------
    T0         : np.ndarray, shape (N, N)  Initial temperature field.
    alpha      : float   Thermal diffusivity.
    dx         : float   Grid spacing.
    n_steps    : int     Number of time steps to run.
    save_every : int     Save the field every this many steps.

    Returns
    -------
    snapshots : np.ndarray, shape (n_saved, N, N)
        Temperature fields at recorded time steps.
    times     : np.ndarray, shape (n_saved,)
        Physical time corresponding to each snapshot.
    dt        : float
        Time step used (derived from stability condition).
    r         : float
        Fourier number used.
    """
    dt, r = compute_stable_dt(dx, alpha)

    print(f"Grid spacing     dx = {dx:.6f}")
    print(f"Time step        dt = {dt:.6f}")
    print(f"Fourier number    r = {r:.4f}  (must be <= 0.25)")
    print(f"Total sim time      = {n_steps * dt:.4f}")

    assert r <= 0.25, f"Unstable! r={r:.4f} exceeds 0.25. Reduce dt or increase dx."

    T         = T0.copy()
    snapshots = []
    times     = []

    for n in range(n_steps + 1):
        if n % save_every == 0:
            snapshots.append(T.copy())
            times.append(n * dt)

        if n < n_steps:
            T = step(T, r)

    return np.array(snapshots), np.array(times), dt, r