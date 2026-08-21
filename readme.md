# Neural Network Surrogate Modeling for 2D Heat Diffusion

**Accuracy, Rollout Stability, and Physics-Informed Training**

[![Python 3.x](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> Companion code for the paper:
> *"Neural Network Surrogate Modeling for 2D Heat Diffusion: Accuracy, Rollout Stability, and Physics-Informed Training"*
> Aditya, 2026.

---

## Overview

This repository contains the complete implementation of a convolutional neural
network (CNN) surrogate model for two-dimensional transient heat diffusion,
including:

- A verified explicit finite-difference solver for the 2D heat equation
- A dataset generation pipeline producing 10,000 supervised training pairs
- A plain CNN surrogate (74,433 parameters) achieving **1.61% median relative L2 error**
- Systematic autoregressive rollout evaluation revealing a **16.52× error growth factor**
- A physics-informed training extension with energy conservation and boundary condition penalties
- Five controlled out-of-distribution generalization experiments
- Four ablation studies (dataset size, model capacity, stride, resolution)

The governing equation is:

```
∂T/∂t = α(∂²T/∂x² + ∂²T/∂y²)
```

on the unit square [0,1]² with homogeneous Dirichlet boundary conditions
(T = 0 on all edges), α = 0.01, discretized on a 64×64 grid.

---

## Key Results

| Metric | Value |
|---|---|
| Single-step RelL2 (median) | **1.61%** |
| Single-step RelL2 (mean) | 2.11% |
| Single-step MAE | 2.25×10⁻³ |
| Rollout error growth (20 steps) | **16.52×** |
| Rollout RelL2 at step 20 | 30.06% |
| Solver effective decay rate λ | +0.115 (decay ✓) |
| CNN effective decay rate λ | −0.198 (growth ✗) |
| BC violations (baseline) | 16/21 rollout steps |
| BC violations (PI λ=0.1) | **0/21** rollout steps |
| Accuracy cost of PI training | +0.093 pp |
| Alpha=0.05 generalization | 28.38% RelL2 |
| Solver time (N=64, 10 steps) | 0.313 ms |
| CNN time (batch=1, CPU) | 5.698 ms |
| Speed crossover resolution | N ≈ 200–256 |

---

## Project Structure

```
heat_surrogate/
│
├── README.md
├── requirements.txt
│
├── src/
│   ├── physics/
│   │   ├── heat_solver.py          # Finite-difference solver
│   │   └── boundary_conditions.py  # Dirichlet and Neumann BC
│   │
│   ├── data/
│   │   ├── generate_dataset.py     # Dataset generation pipeline
│   │   └── dataset.py              # PyTorch Dataset wrapper
│   │
│   ├── models/
│   │   ├── cnn.py                  # Baseline CNN surrogate
│   │   └── physics_informed.py     # Physics-informed loss
│   │
│   ├── training/
│   │   ├── train.py                # Baseline training loop
│   │   └── train_physics_informed.py  # PI training loop
│   │
│   └── evaluation/
│       ├── metrics.py              # MSE, MAE, RelL2, MaxAE
│       └── benchmark.py            # Speed benchmarking
│
├── notebooks/
│   ├── milestone2_solver.ipynb         # Solver validation
│   ├── milestone3_dataset.ipynb        # Dataset generation
│   ├── milestone4_eda.ipynb            # Exploratory data analysis
│   ├── milestone5_training.ipynb       # Model training
│   ├── milestone6_evaluation.ipynb     # Scientific evaluation
│   ├── milestone7_benchmark.ipynb      # Speed comparison
│   ├── milestone8_rollout.ipynb        # Rollout stability
│   ├── milestone9_generalization.ipynb # OOD generalization
│   ├── milestone10_ablation.ipynb      # Ablation studies
│   └── milestone11_physics_informed.ipynb  # PI training
│
├── data/                    # Generated dataset (see below)
├── results/                 # Model checkpoints and JSON results
└── figures/                 # All paper figures
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Aditya30ag/heat-surrogate.git
cd heat-surrogate
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt:**

```
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
torch>=2.0.0
jupyter>=1.0.0
```

---

## Quick Start

### Step 1 — Generate the dataset

```bash
python src/data/generate_dataset.py
```

This generates 10,000 supervised (input, target) pairs using the
finite-difference solver and saves them to `data/heat_dataset.npz`.

Expected output:
```
Generating sample      0 / 10000 ...
Generating sample   1000 / 10000 ...
...
✓ All sanity checks passed.
✓ Dataset saved to: data/heat_dataset.npz
  File size: ~150 MB (compressed)

Split summary:
  Train : (8000, 64, 64)
  Val   : (1000, 64, 64)
  Test  : (1000, 64, 64)
```

### Step 2 — Train the baseline CNN

```bash
python src/training/train.py \
    --dataset_path data/heat_dataset.npz \
    --save_dir results \
    --n_filters 32 \
    --batch_size 32 \
    --n_epochs 100 \
    --lr 1e-3 \
    --seed 42
```

Expected result:
```
Best epoch       : 19
Best val MSE     : 7.28e-06
Total parameters : 74,433
```

### Step 3 — Evaluate on the test set

```bash
python src/evaluation/metrics.py \
    --dataset_path data/heat_dataset.npz \
    --checkpoint results/best_model.pt
```

Expected result:
```
Relative L2 Error:
  Mean   : 2.107%
  Median : 1.613%
  Std    : 1.375%
  95th % : 4.814%
```

### Step 4 — Run the speed benchmark

```bash
python src/evaluation/benchmark.py \
    --checkpoint results/best_model.pt
```

### Step 5 — Train a physics-informed model

```bash
python src/training/train_physics_informed.py \
    --dataset_path data/heat_dataset.npz \
    --save_dir results \
    --lambda_energy 0.1 \
    --lambda_bc 0.1 \
    --run_name pi_lambda_010 \
    --n_epochs 100 \
    --seed 42
```

---

## Reproducing Paper Results

Run all notebooks in order to reproduce every figure and table in the paper:

| Notebook | Paper section | Key output |
|---|---|---|
| `milestone2_solver.ipynb` | Section 4 | Solver validation plots |
| `milestone3_dataset.ipynb` | Section 4 | Dataset generation |
| `milestone4_eda.ipynb` | Section 4 | EDA figures |
| `milestone5_training.ipynb` | Section 5 | Training curves |
| `milestone6_evaluation.ipynb` | Section 6.1 | Table 1, Figures 1–2 |
| `milestone7_benchmark.ipynb` | Section 6.2 | Table 2, Figure 3 |
| `milestone8_rollout.ipynb` | Section 6.3 | Table 3, Figures 4–6 |
| `milestone9_generalization.ipynb` | Section 6.4 | Tables 4–7, Figure 7 |
| `milestone10_ablation.ipynb` | Section 6.5 | Tables 8–11, Figure 8 |
| `milestone11_physics_informed.ipynb` | Section 6.6 | Tables 12–15, Figures 9–10 |

---

## Architecture

The CNN surrogate has the following structure:

```
Input  (B, 1, 64, 64)
  → ConvBlock(1  → 32):  Conv2d(k=3) + BatchNorm + ReLU
  → ConvBlock(32 → 64):  Conv2d(k=3) + BatchNorm + ReLU
  → ConvBlock(64 → 64):  Conv2d(k=3) + BatchNorm + ReLU
  → ConvBlock(64 → 32):  Conv2d(k=3) + BatchNorm + ReLU
  → Conv2d(32 → 1, k=1)  [no activation]
Output (B, 1, 64, 64)

Total trainable parameters: 74,433
```

All convolutions use `padding=1` to preserve spatial dimensions.
The output layer has no activation — temperature is physically unbounded.

---

## Physics-Informed Loss

The physics-informed total loss is:

```
L_total = L_data + λ_en * L_energy + λ_bc * L_bc
```

where:

- `L_data` = MSE against solver ground truth
- `L_energy` = ReLU(mean(pred) − mean(input)) — penalizes heat retention
- `L_bc` = mean squared boundary values — penalizes BC violations
- `λ_en = λ_bc = λ_phys` ∈ {0.01, 0.05, 0.10}

Results at λ=0.10:

| Property | Baseline | PI (λ=0.10) |
|---|---|---|
| Single-step RelL2 | 2.107% | 2.200% (+0.093pp) |
| Decay direction | Growth ✗ | Decay ✓ |
| BC violations | 16/21 | 0/21 |
| Step-20 mean temp | 163% of solver | 34% of solver |

---

## Key Findings

### Finding 1 — Decay Rate Sign Reversal

A CNN achieving 1.61% single-step accuracy produces an effective
thermal decay rate of **λ = −0.198** during rollout, versus
**λ = +0.115** for the true solver — a sign reversal from
physical decay to unphysical heat growth. This arises from a
1.45% per-step heat retention bias that compounds geometrically:

```
1.0145^20 ≈ 1.335  →  33% mean temperature excess at step 20
```

### Finding 2 — Metric Normalization Artifact

Out-of-distribution samples with 5 blobs show **lower** relative L2
error (1.29%) than the in-distribution baseline (2.19%), despite no
genuine improvement in absolute prediction quality. This occurs because
more blobs produce larger ‖T‖₂ denominators, reducing the relative
percentage regardless of absolute error magnitude. Always report MAE
alongside relative L2 when comparing test conditions with different
signal magnitudes.

### Finding 3 — Over-Regularization in PI Training

Asymmetric ReLU-based energy penalties correct the decay direction
but cause over-dissipation. At λ=0.10, step-20 mean temperature
falls to 34% of the true solver value (physical target: 82%).
The mechanism: ReLU penalizes energy increase but places no lower
bound on energy decrease. A symmetric MSE formulation targeting
the ground-truth mean is proposed as future work.

---

## Experimental Configuration

| Parameter | Value |
|---|---|
| Domain | [0,1]² |
| Grid resolution | 64×64 |
| Grid spacing Δx | 1/63 ≈ 0.0159 |
| Thermal diffusivity α | 0.01 |
| Boundary conditions | Dirichlet (T=0) |
| Solver time step Δt | 0.005669 |
| Fourier number r | 0.225 (< 0.25 ✓) |
| Prediction stride s | 10 solver steps |
| CNN prediction window | 0.0567 physical time |
| Training samples | 8,000 |
| Validation samples | 1,000 |
| Test samples | 1,000 |
| Random seed | 42 |

---

## Citation

If you use this code or findings in your research, please cite:

```bibtex
@article{aditya2026heatsurrogate,
  author  = {Aditya},
  title   = {Neural Network Surrogate Modeling for 2D Heat Diffusion:
             Accuracy, Rollout Stability, and Physics-Informed Training},
  journal = {[Journal name]},
  year    = {2026},
  url     = {https://github.com/Aditya30ag/heat_surrogate.git}
}
```

---

## License

This project is licensed under the MIT License.
See [LICENSE](LICENSE) for details.

---

## Contact

For questions about this work, please open a GitHub issue or
contact the author at `adityaagrwal3005@gmail.com`.