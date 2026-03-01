# TNAFMOL — Environment Setup

## Current Machine
- **OS:** Windows 11 Pro 10.0.26200
- **Python:** 3.11.9 (Microsoft Store version)
- **GPU:** NVIDIA RTX 2000 Ada Generation, 8GB VRAM
- **CUDA:** 13.0 (driver 581.60)
- **Shell:** bash (Git Bash / MSYS2)

## Environment Issues (as of 2026-02-28)
- PyTorch is installed but broken (DLL load failure on `torch._C`). Needs reinstall.
- Missing packages: wandb, h5py, ase, torchani, rdkit
- Available: numpy 2.3.5, scipy 1.16.3, matplotlib 3.10.7

## Required Packages
- **torch** (with CUDA support for CUDA 13.0)
- **numpy**, **scipy**, **matplotlib** (already installed)
- **h5py** (for HDF5 dataset storage)
- **wandb** (W&B logging)
- **ase** (Atomic Simulation Environment — for molecular data handling)
- **torchani** or similar (for ANI-2x energy evaluation)

## Setup Steps
*(To be filled by the agent that fixes the environment.)*
