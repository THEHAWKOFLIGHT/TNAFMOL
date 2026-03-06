"""
train_phase3.py — Phase 3 Adaptation Ladder for und_001 TarFlow Diagnostic.

Runs Steps A through F, each training from scratch for 5000 steps on ethanol.
Each step adds ONE change to the Apple TarFlow architecture.

PURPOSE: Identify which architectural change degrades performance (log-det exploitation).
HYPOTHESIS: Step E (shared scale) is the break point.

Steps:
  A: Raw atomic coordinates — pure Apple TarFlow1D on 9 ethanol atoms
  B: Add atom type conditioning (nn.Embedding 4->16, concat to input)
  C: Add padding + attention masking (21 atoms, causal+padding mask)
  D: Add Gaussian noise augmentation (sigma=0.05, real atoms only)
  E: Switch to shared scale (1 scalar per atom, 3x log-det leverage) — KEY TEST
  F: Add stabilization (asymmetric clamp + log-det regularization)

Usage:
  # Run single step on GPU 5 (logical device 0 when CUDA_VISIBLE_DEVICES=5,6)
  CUDA_VISIBLE_DEVICES=5 python src/train_phase3.py --step a --gpu 0
  CUDA_VISIBLE_DEVICES=6 python src/train_phase3.py --step b --gpu 0

  # Run all steps
  CUDA_VISIBLE_DEVICES=5 python src/train_phase3.py --step all --gpu 0
"""

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))
from tarflow_apple import (
    TarFlow1D, MetaBlock, Permutation, PermutationIdentity, PermutationFlip,
    Attention, MLP, AttentionBlock,
)
from data import MD17Dataset
import metrics as metrics_module


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ETHANOL_DATA_DIR = 'data/md17_ethanol_v1'
N_REAL_ATOMS = 9        # ethanol has 9 atoms
N_PADDED_ATOMS = 21     # full padding for compatibility with src/model.py convention
ATOM_EMB_DIM = 16       # embedding dimension for atom types
NUM_ATOM_TYPES = 4      # H, C, N, O
RESULTS_BASE = 'experiments/understanding/und_001_tarflow_diagnostic/results/phase3'

STEP_DIRS = {
    'a': 'step_a_raw_coords',
    'b': 'step_b_atom_type',
    'c': 'step_c_padding_mask',
    'd': 'step_d_noise_aug',
    'e': 'step_e_shared_scale',
    'f': 'step_f_stabilization',
}

STEP_DESCRIPTIONS = {
    'a': 'Raw atomic coordinates — pure Apple TarFlow1D on 9 ethanol atoms (no conditioning, no padding)',
    'b': 'Atom type conditioning — nn.Embedding(4, 16) concat to input per atom',
    'c': 'Padding + attention masking — 21 atoms, causal+padding attention mask',
    'd': 'Gaussian noise augmentation — sigma=0.05 to real atoms during training',
    'e': 'Shared scale — 1 scalar per atom applied to all 3 coords (KEY TEST for log-det exploitation)',
    'f': 'Stabilization — asymmetric clamp (alpha_pos=0.1) + log-det regularization (weight=0.01)',
}


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# ---------------------------------------------------------------------------
# MetaBlock with shared scale (Step E and F)
# ---------------------------------------------------------------------------

class MetaBlockSharedScale(nn.Module):
    """
    Flow block variant with SHARED scale: ONE scalar per atom applied to all 3 coords.

    Key difference from Apple MetaBlock:
    - Apple (per-dim): xa shape (B, T, 3) — 3 independent scales per atom
      log_det = -xa.mean([1,2])  (averages over T and D=3)
    - Here (shared): xa shape (B, T, 1) — single shared scale per atom
      log_det = -xa.squeeze(-1).mean([1]) — but when we apply exp(-xa) to 3 coords,
      the actual change-of-variables formula is:
      log|det J| = sum over atoms of (-3 * s_i * mask_i) / T / D
      For a batch: logdet = -3 * xa.squeeze(-1).mean([1])  [per sample, averaged over atoms]

    This is the KEY TEST: with shared scale, each unit of log_scale exploited gives
    3x the log_det benefit compared to per-dimension scale (since same scalar × 3 coords).
    This 3x leverage concentrates the optimizer's incentive to exploit log_det.
    """
    attn_mask: torch.Tensor

    def __init__(
        self,
        in_channels: int,       # = 3 (position dims)
        channels: int,
        num_patches: int,
        permutation: Permutation,
        num_layers: int = 1,
        head_dim: int = 64,
        expansion: int = 4,
        cond_channels: int = 0,     # additional conditioning channels (atom type embeddings)
        use_clamp: bool = False,    # Step F: asymmetric clamping
        alpha_pos: float = 0.1,
        alpha_neg: float = 2.0,
    ):
        super().__init__()
        # proj_in takes (in_channels + cond_channels) -> channels
        self.proj_in = nn.Linear(in_channels + cond_channels, channels)
        self.pos_embed = nn.Parameter(torch.randn(num_patches, channels) * 1e-2)

        self.attn_blocks = nn.ModuleList(
            [AttentionBlock(channels, head_dim, expansion) for _ in range(num_layers)]
        )

        # Shared scale: output 1 scale + in_channels shift = in_channels + 1
        output_dim = 1 + in_channels  # (shared scale, shift_x, shift_y, shift_z)
        self.proj_out = nn.Linear(channels, output_dim)
        self.proj_out.weight.data.fill_(0.0)

        self.permutation = permutation
        self.in_channels = in_channels
        self.cond_channels = cond_channels
        self.use_clamp = use_clamp
        self.alpha_pos = alpha_pos
        self.alpha_neg = alpha_neg

        self.register_buffer(
            'attn_mask', torch.tril(torch.ones(num_patches, num_patches))
        )

    def _clamp(self, s: torch.Tensor) -> torch.Tensor:
        """Asymmetric soft clamping (Andrade et al. 2024)."""
        if not self.use_clamp:
            return s
        pos_mask = (s >= 0).float()
        neg_mask = 1.0 - pos_mask
        clamped = (
            pos_mask * (2.0 / math.pi) * self.alpha_pos * torch.atan(s / self.alpha_pos)
            + neg_mask * (2.0 / math.pi) * self.alpha_neg * torch.atan(s / self.alpha_neg)
        )
        return clamped

    def forward(
        self,
        x: torch.Tensor,            # (B, T, 3) positions
        cond: torch.Tensor | None = None,  # (B, T, cond_channels) atom type embeddings
        padding_mask: torch.Tensor | None = None,  # (B, T) bool: True = real atom
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, 3) input positions
            cond: (B, T, cond_channels) optional conditioning
            padding_mask: (B, T) True for real atoms — used to zero out padding contributions

        Returns:
            z: (B, T, 3) latent
            logdet: (B,) per-sample log-determinant contribution
        """
        x = self.permutation(x)
        pos_embed = self.permutation(self.pos_embed, dim=0)

        x_in = x  # save original for affine transform (already permuted)

        # CRITICAL: permute the padding mask to match the permuted token order.
        # PermutationFlip reverses tokens, so mask must also be reversed.
        # Without this, PermutationFlip blocks zero the wrong positions.
        mask_perm = None
        if padding_mask is not None:
            mask_perm = self.permutation(
                padding_mask.float().unsqueeze(-1)  # (B, T, 1)
            ).squeeze(-1)  # (B, T)

        # Build input to proj_in: positions [+ conditioning]
        if cond is not None and self.cond_channels > 0:
            cond_perm = self.permutation(cond)
            x_proj = torch.cat([x, cond_perm], dim=-1)  # (B, T, 3 + cond_channels)
        else:
            x_proj = x

        x_hidden = self.proj_in(x_proj) + pos_embed  # (B, T, channels)

        # Build causal attention mask (in permuted space), combined with permuted padding mask
        attn_mask = self.attn_mask  # (T, T) lower triangular
        if mask_perm is not None:
            B = x.size(0)
            pad_key_mask = mask_perm.unsqueeze(1).expand(B, attn_mask.size(0), -1)  # (B, T, T)
            attn_mask_full = (attn_mask.unsqueeze(0) * pad_key_mask).unsqueeze(1)  # (B, 1, T, T)
        else:
            attn_mask_full = attn_mask  # (T, T) will broadcast

        # Attention blocks
        for block in self.attn_blocks:
            x_hidden = block(x_hidden, attn_mask_full)

        x_out = self.proj_out(x_hidden)  # (B, T, 1 + 3)

        # Output shift: auto-regressive shift (in permuted space)
        x_out = torch.cat([torch.zeros_like(x_out[:, :1]), x_out[:, :-1]], dim=1)

        # Split: xa = shared scale (B, T, 1), xb = shift (B, T, 3)
        xa = x_out[..., :1]     # (B, T, 1) — shared log_scale per atom
        xb = x_out[..., 1:]     # (B, T, 3) — shift per atom per coord

        # Zero out affine params for padding atoms in PERMUTED space.
        # mask_perm has 1s at real atom positions (in permuted order), 0s at padding.
        # Forces padding atoms to identity: z_pad = (0 - 0) * exp(0) = 0.
        if mask_perm is not None:
            pad_float = mask_perm.unsqueeze(-1)  # (B, T, 1)
            xa = xa * pad_float   # (B, T, 1): zero scale for padding
            xb = xb * pad_float   # (B, T, 3): zero shift for padding

        # Apply clamping to shared scale
        xa = self._clamp(xa)

        # Affine forward: z = (x - shift) * exp(-shared_scale)
        # xa is (B, T, 1), broadcasted to (B, T, 3)
        scale = (-xa.float()).exp().type(xa.dtype)  # (B, T, 1)
        z = (x_in - xb) * scale  # broadcasts: scale (B,T,1) × (B,T,3)
        # z for padding atoms is now exactly 0: (0 - 0) * exp(0) = 0

        z = self.permutation(z, inverse=True)

        # Log-determinant: each atom contributes -3 * xa_i to log|det J| (3 coords scaled by xa)
        # The forward transform is z_ij = (x_ij - b_ij) * exp(-s_i) for j in {x,y,z}
        # log|det J_i| = -3 * s_i (same scalar for all 3 coords)
        # Normalize by T (num_patches) and D=3 coords (matching Apple's normalization):
        # Apple: logdet = -xa.mean([1,2])  → averages over T tokens AND D dims
        # Shared: sum over atoms of (-3 * s_i) / (T * D) = -xa.squeeze(-1).mean(1)
        # (The factor of 3 cancels with D=3 normalization — same formula as Apple per-dim.)
        #
        # Padding atoms: xa was zeroed above, so their logdet contribution is 0.
        # Normalize by n_real (NOT T) to match the z² reconstruction normalization.
        # If we normalize by T, the logdet gradient per real atom is scaled by n_real/T,
        # shifting the equilibrium outside the data distribution — causing log-det exploitation.
        if padding_mask is not None:
            n_real = padding_mask.float().sum(dim=1)  # (B,) number of real atoms per sample
            logdet = -xa.squeeze(-1).sum(dim=1) / n_real  # (B,): sum over real atoms / n_real
        else:
            logdet = -xa.squeeze(-1).mean(dim=1)  # (B,): mean over T

        return z, logdet

    def reverse_step(
        self,
        x: torch.Tensor,
        pos_embed: torch.Tensor,
        i: int,
        cond: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One autoregressive step during sampling."""
        x_in = x[:, i:i + 1]  # (B, 1, 3)

        if cond is not None and self.cond_channels > 0:
            cond_i = cond[:, i:i + 1]  # (B, 1, cond_channels)
            x_proj = torch.cat([x_in, cond_i], dim=-1)
        else:
            x_proj = x_in

        x_hidden = self.proj_in(x_proj) + pos_embed[i:i + 1]

        for block in self.attn_blocks:
            x_hidden = block(x_hidden)

        x_out = self.proj_out(x_hidden)

        xa = x_out[..., :1]   # (B, 1, 1) shared scale
        xb = x_out[..., 1:]   # (B, 1, 3) shift
        xa = self._clamp(xa)

        return xa, xb

    def set_sample_mode(self, flag: bool = True):
        for m in self.modules():
            if isinstance(m, Attention):
                m.sample = flag
                m.k_cache = {'cond': [], 'uncond': []}
                m.v_cache = {'cond': [], 'uncond': []}

    def reverse(
        self,
        x: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Reverse pass (latent → data)."""
        x = self.permutation(x)
        pos_embed = self.permutation(self.pos_embed, dim=0)

        if cond is not None:
            cond = self.permutation(cond)

        self.set_sample_mode(True)
        T = x.size(1)

        for i in range(T - 1):
            za, zb = self.reverse_step(x, pos_embed, i, cond)
            # Inverse: x[i+1] = z[i+1] * exp(xa) + xb
            scale = za[:, 0].float().exp().type(za.dtype)  # (B, 1)
            x[:, i + 1] = x[:, i + 1] * scale + zb[:, 0]

        self.set_sample_mode(False)
        return self.permutation(x, inverse=True)


# ---------------------------------------------------------------------------
# TarFlow1DMol — with atom type conditioning and optional padding/shared scale
# ---------------------------------------------------------------------------

class TarFlow1DMol(nn.Module):
    """
    TarFlow1D variant for molecular data.

    Extends TarFlow1D with:
    - Atom type conditioning via learned embedding
    - Optional padding + attention masking
    - Optional shared scale (for Step E/F)
    - Optional stabilization (clamping + log-det reg)
    """

    def __init__(
        self,
        in_channels: int,        # 3 (positions)
        seq_length: int,         # 9 (real atoms) or 21 (padded)
        channels: int,
        num_blocks: int,
        layers_per_block: int,
        head_dim: int = 64,
        expansion: int = 4,
        # Conditioning
        use_atom_type_cond: bool = False,
        atom_type_emb_dim: int = 16,
        num_atom_types: int = 4,
        # Padding
        use_padding_mask: bool = False,
        # Shared scale
        use_shared_scale: bool = False,
        # Stabilization (Step F)
        use_clamp: bool = False,
        alpha_pos: float = 0.1,
        alpha_neg: float = 2.0,
        log_det_reg_weight: float = 0.0,
    ):
        super().__init__()
        self.seq_length = seq_length
        self.in_channels = in_channels
        self.use_atom_type_cond = use_atom_type_cond
        self.use_padding_mask = use_padding_mask
        self.use_shared_scale = use_shared_scale
        self.log_det_reg_weight = log_det_reg_weight

        # Atom type embedding
        if use_atom_type_cond:
            self.atom_emb = nn.Embedding(num_atom_types, atom_type_emb_dim)
            cond_channels = atom_type_emb_dim
        else:
            self.atom_emb = None
            cond_channels = 0

        permutations = [
            PermutationIdentity(seq_length),
            PermutationFlip(seq_length),
        ]

        blocks = []
        for i in range(num_blocks):
            if use_shared_scale:
                # Step E/F: shared scale variant
                blocks.append(
                    MetaBlockSharedScale(
                        in_channels=in_channels,
                        channels=channels,
                        num_patches=seq_length,
                        permutation=permutations[i % 2],
                        num_layers=layers_per_block,
                        head_dim=head_dim,
                        expansion=expansion,
                        cond_channels=cond_channels,
                        use_clamp=use_clamp,
                        alpha_pos=alpha_pos,
                        alpha_neg=alpha_neg,
                    )
                )
            else:
                # Steps A-D: Apple MetaBlock with optional conditioning
                if use_atom_type_cond:
                    # Extend proj_in to accept in_channels + cond_channels
                    # We create a MetaBlock then replace proj_in
                    block = MetaBlock(
                        in_channels=in_channels + cond_channels,  # proj_in sees both
                        channels=channels,
                        num_patches=seq_length,
                        permutation=permutations[i % 2],
                        num_layers=layers_per_block,
                        head_dim=head_dim,
                        expansion=expansion,
                        nvp=True,
                    )
                    # But the affine params should only act on in_channels=3 positions.
                    # The proj_out output size is (in_channels + cond_channels) * 2.
                    # This is WRONG — we want to only transform the 3D positions.
                    # Correct approach: proj_out produces in_channels * 2 = 6 dims.
                    # Fix: replace proj_out with correct output dim.
                    output_dim = in_channels * 2
                    block.proj_out = nn.Linear(channels, output_dim)
                    block.proj_out.weight.data.fill_(0.0)
                    # Also need to fix: MetaBlock was built with in_channels = 3+emb_dim
                    # but proj_out should produce 3*2=6 dims, and the affine acts on 3 dims.
                    # Wrap block to handle this.
                    blocks.append(MetaBlockWithCond(block, in_channels, cond_channels))
                else:
                    blocks.append(
                        MetaBlock(
                            in_channels=in_channels,
                            channels=channels,
                            num_patches=seq_length,
                            permutation=permutations[i % 2],
                            num_layers=layers_per_block,
                            head_dim=head_dim,
                            expansion=expansion,
                            nvp=True,
                        )
                    )

        self.blocks = nn.ModuleList(blocks)
        self.register_buffer('var', torch.ones(seq_length, in_channels))

    def forward(
        self,
        x: torch.Tensor,            # (B, T, 3) positions
        atom_types: torch.Tensor | None = None,  # (B, T) or (T,) integer atom type indices
        padding_mask: torch.Tensor | None = None,  # (B, T) float 1=real, 0=pad
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns:
            z: (B, T, 3) latent
            logdets: (B,) accumulated log-determinant per sample
        """
        # Compute atom type conditioning
        cond = None
        if self.use_atom_type_cond and atom_types is not None and self.atom_emb is not None:
            if atom_types.dim() == 1:
                atom_types = atom_types.unsqueeze(0).expand(x.size(0), -1)
            cond = self.atom_emb(atom_types)  # (B, T, emb_dim)

        # Convert padding_mask to bool for attention
        pad_bool = None
        if self.use_padding_mask and padding_mask is not None:
            pad_bool = padding_mask.bool()  # (B, T)

        logdets = torch.zeros(x.size(0), device=x.device)
        for block in self.blocks:
            if self.use_shared_scale:
                x, logdet = block(x, cond=cond, padding_mask=pad_bool)
            elif self.use_atom_type_cond:
                x, logdet = block(x, cond=cond, padding_mask=pad_bool)
            else:
                # Pure Apple blocks (Steps A-C without conditioning)
                if self.use_padding_mask and pad_bool is not None:
                    x, logdet = block(x, attn_mask_extra=pad_bool)
                else:
                    x, logdet = block(x)
            logdets = logdets + logdet

        return x, logdets

    def get_loss(
        self,
        z: torch.Tensor,
        logdets: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        NLL per dimension: 0.5 * mean(z²) - mean(logdets)

        For padded inputs, only averages over real atom positions.

        Returns:
            loss: scalar NLL
            info: dict with component values for tracking
        """
        if padding_mask is not None:
            # Only compute NLL over real atoms
            mask_3d = padding_mask.unsqueeze(-1).float()  # (B, T, 1) broadcast to (B, T, 3)
            n_real_per_sample = padding_mask.float().sum(1)  # (B,)
            n_real_total = (n_real_per_sample * z.size(-1)).mean()  # average n_dof per sample

            # z^2 over real atoms
            z_sq = (z ** 2 * mask_3d).sum([1, 2]) / (n_real_per_sample * z.size(-1))  # (B,)
            nll = 0.5 * z_sq.mean() - logdets.mean()
        else:
            nll = 0.5 * z.pow(2).mean() - logdets.mean()

        # Log-det regularization (Step F only)
        loss = nll
        reg_val = 0.0
        if self.log_det_reg_weight > 0:
            if padding_mask is not None:
                n_real_per_sample = padding_mask.float().sum(1)  # (B,)
                n_dof = n_real_per_sample * z.size(-1)  # (B,)
                logdet_per_dof = logdets / n_dof.clamp(min=1.0)
            else:
                n_dof = z.size(1) * z.size(2)
                logdet_per_dof = logdets / n_dof
            reg = self.log_det_reg_weight * (logdet_per_dof ** 2).mean()
            loss = nll + reg
            reg_val = reg.item()

        info = {
            'nll': nll.item(),
            'reg': reg_val,
            'logdets_mean': logdets.mean().item(),
        }
        return loss, info

    @torch.no_grad()
    def reverse(
        self,
        x: torch.Tensor,
        atom_types: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Reverse pass (latent → data).

        CRITICAL: If padding_mask is provided, padding positions in x must already
        be zeroed before calling this method (done in sample()). The padding zeros
        are maintained between block calls to prevent padding noise from corrupting
        real atom generation in PermutationFlip blocks.

        In PermutationFlip blocks, the sequence is reversed so padding atoms appear
        at the BEGINNING of the autoregressive chain. If padding positions contain
        nonzero noise, that noise propagates into all subsequent real atom positions.
        Zeroing padding positions in the latent and between blocks matches the
        forward pass where z_pad = 0 exactly (enforced by mask_perm zeroing).
        """
        x = x * self.var.sqrt()

        # Zero out padding positions immediately after scaling.
        # This matches the forward pass where z_pad = 0 for all padding atoms.
        # Without this, padding Gaussian noise propagates into real atom generation
        # through the autoregressive chain in PermutationFlip blocks.
        if self.use_padding_mask and padding_mask is not None:
            mask_3d = padding_mask.float().unsqueeze(-1)
            x = x * mask_3d

        cond = None
        if self.use_atom_type_cond and atom_types is not None and self.atom_emb is not None:
            if atom_types.dim() == 1:
                atom_types = atom_types.unsqueeze(0).expand(x.size(0), -1)
            cond = self.atom_emb(atom_types)

        for block in reversed(self.blocks):
            if self.use_shared_scale:
                x = block.reverse(x, cond=cond)
            elif self.use_atom_type_cond:
                x = block.reverse(x, cond=cond)
            else:
                x = block.reverse(x)

            # Re-zero padding positions after each block.
            # Each block's reverse() uses the full T-length tensor as KV cache input,
            # so padding positions in the output of one block become inputs to the next.
            # Zeroing between blocks keeps the padding zone clean throughout the chain.
            if self.use_padding_mask and padding_mask is not None:
                x = x * mask_3d

        return x

    @torch.no_grad()
    def sample(
        self,
        n: int,
        device: torch.device,
        atom_types: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        temp: float = 1.0,
    ) -> torch.Tensor:
        """Sample n sequences.

        Padding positions in the latent z are zeroed before the reverse pass
        to match the forward pass convention (z_pad = 0 exactly).
        """
        z = torch.randn(n, self.seq_length, self.in_channels, device=device) * temp

        # Zero padding positions in latent immediately — before any reverse pass.
        # This ensures PermutationFlip blocks see zeros at padding positions,
        # which is what was learned during training (z_pad = 0 in forward).
        if self.use_padding_mask and padding_mask is not None:
            mask_3d = padding_mask.float().unsqueeze(-1)
            z = z * mask_3d

        return self.reverse(z, atom_types=atom_types, padding_mask=padding_mask)


# ---------------------------------------------------------------------------
# MetaBlockWithCond — wrapper around MetaBlock for atom type conditioning (Steps B-D)
# ---------------------------------------------------------------------------

class MetaBlockWithCond(nn.Module):
    """
    Wraps Apple MetaBlock to support atom type conditioning.

    Strategy: concatenate atom type embeddings to positions before proj_in,
    but the affine transform still only acts on the 3 position dimensions.
    proj_out produces in_channels * 2 = 6 dims (xa=3 per-dim scales, xb=3 shifts).

    This is the cleanest approach: the transformer sees [pos | emb] as input,
    but the flow coupling only transforms the 3 position coordinates.
    """

    def __init__(self, base_block: MetaBlock, in_channels: int, cond_channels: int):
        super().__init__()
        self.base = base_block
        self.in_channels = in_channels
        self.cond_channels = cond_channels
        # base_block was built with in_channels = in_channels + cond_channels (for proj_in size)
        # base_block.proj_out was re-sized to in_channels * 2

    def forward(
        self,
        x: torch.Tensor,        # (B, T, 3)
        cond: torch.Tensor | None = None,   # (B, T, cond_channels)
        padding_mask: torch.Tensor | None = None,  # (B, T) bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass. Concatenates cond to x before passing to the base MetaBlock.
        The base MetaBlock's affine transform operates on in_channels=3 positions only.
        """
        # Apply permutation to x and cond consistently
        x_perm = self.base.permutation(x)
        if cond is not None:
            cond_perm = self.base.permutation(cond)
        else:
            cond_perm = None

        # CRITICAL: permute the padding mask to match the permuted token order.
        # PermutationFlip reverses token order, so mask must also be reversed.
        # Without this, PermutationFlip blocks zero the wrong positions (real instead of padding).
        mask_perm = None
        if padding_mask is not None:
            # padding_mask: (B, T) bool — apply same permutation as x
            # Use unsqueeze/squeeze trick to apply permutation along dim=1 of a 2D tensor
            mask_perm = self.base.permutation(
                padding_mask.float().unsqueeze(-1)  # (B, T, 1)
            ).squeeze(-1)  # (B, T)

        # Build combined attention mask (in permuted space)
        attn_mask = self.base.attn_mask  # (T, T)
        if mask_perm is not None:
            B = x.size(0)
            # Key masking in permuted space: query i can only attend to real key j
            pad_key = mask_perm.unsqueeze(1).expand(B, attn_mask.size(0), -1)  # (B, T, T)
            attn_mask_combined = (attn_mask.unsqueeze(0) * pad_key).unsqueeze(1)  # (B, 1, T, T)
        else:
            attn_mask_combined = attn_mask  # (T, T) — broadcasts correctly

        # Concatenate conditioning to positions for proj_in
        if cond_perm is not None:
            x_for_proj = torch.cat([x_perm, cond_perm], dim=-1)  # (B, T, 3+emb)
        else:
            x_for_proj = x_perm  # shouldn't happen but handle gracefully

        # Project and add positional embeddings
        pos_embed = self.base.permutation(self.base.pos_embed, dim=0)
        x_hidden = self.base.proj_in(x_for_proj) + pos_embed

        # Apply attention blocks
        for block in self.base.attn_blocks:
            x_hidden = block(x_hidden, attn_mask_combined)

        x_out = self.base.proj_out(x_hidden)  # (B, T, 6) — 3 scale + 3 shift

        # Output shift for autoregression (in permuted space)
        x_out = torch.cat([torch.zeros_like(x_out[:, :1]), x_out[:, :-1]], dim=1)

        # NVP split: xa = (B, T, 3) log-scale per dim, xb = (B, T, 3) shift
        xa, xb = x_out.chunk(2, dim=-1)

        # Zero out affine params for padding atoms BEFORE applying transform (in permuted space).
        # mask_perm is the padding mask in permuted order — same order as xa, xb, x_perm.
        # Forces padding atoms to identity: z_pad = (0 - 0) * exp(0) = 0.
        if mask_perm is not None:
            pad_float = mask_perm.unsqueeze(-1)  # (B, T, 1)
            xa = xa * pad_float   # (B, T, 3): zero log-scale for padding
            xb = xb * pad_float   # (B, T, 3): zero shift for padding

        # Affine transform on positions only
        scale = (-xa.float()).exp().type(xa.dtype)
        z = (x_perm - xb) * scale
        # z for padding atoms: (0 - 0) * exp(0) = 0 exactly

        z = self.base.permutation(z, inverse=True)

        # Log-det: xa zeroed for padding atoms, so -xa.sum = -sum over real atoms only.
        # Normalize by n_real * D (matching get_loss per-real-dof normalization).
        if padding_mask is not None:
            n_real = padding_mask.float().sum(dim=1)  # (B,) number of real atoms per sample
            logdet = -xa.sum(dim=[1, 2]) / (n_real * xa.size(-1))  # (B,)
        else:
            logdet = -xa.mean(dim=[1, 2])  # (B,)

        return z, logdet

    def reverse(
        self,
        x: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Reverse pass using KV-cached autoregressive sampling."""
        x_perm = self.base.permutation(x)
        pos_embed = self.base.permutation(self.base.pos_embed, dim=0)

        if cond is not None:
            cond_perm = self.base.permutation(cond)
        else:
            cond_perm = None

        # Enable KV caching
        for m in self.base.modules():
            if isinstance(m, Attention):
                m.sample = True
                m.k_cache = {'cond': [], 'uncond': []}
                m.v_cache = {'cond': [], 'uncond': []}

        T = x_perm.size(1)
        for i in range(T - 1):
            x_in = x_perm[:, i:i + 1]  # (B, 1, 3)

            if cond_perm is not None:
                cond_i = cond_perm[:, i:i + 1]
                x_proj = torch.cat([x_in, cond_i], dim=-1)
            else:
                x_proj = x_in

            x_hidden = self.base.proj_in(x_proj) + pos_embed[i:i + 1]
            for block in self.base.attn_blocks:
                x_hidden = block(x_hidden)

            x_out = self.base.proj_out(x_hidden)
            xa, xb = x_out.chunk(2, dim=-1)

            scale = xa[:, 0].float().exp().type(xa.dtype)  # (B, 3)
            x_perm[:, i + 1] = x_perm[:, i + 1] * scale + xb[:, 0]

        # Disable KV caching
        for m in self.base.modules():
            if isinstance(m, Attention):
                m.sample = False

        return self.base.permutation(x_perm, inverse=True)


# ---------------------------------------------------------------------------
# MetaBlockPaddingMask — Step C: Apple MetaBlock with padding + causal mask
# ---------------------------------------------------------------------------

class MetaBlockWithPadding(nn.Module):
    """
    Apple MetaBlock augmented with padding mask support (Step C).

    No atom type conditioning here (that comes in step B; step C adds padding on top of B).
    Steps C-D use MetaBlockWithCond which already handles padding mask.
    This class handles the edge case of step A+C (padding only, no conditioning) — not needed.
    Kept for reference / completeness.
    """
    pass  # Not instantiated directly — MetaBlockWithCond handles all cases


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_ethanol(data_dir: str, seq_length: int = 9) -> tuple:
    """
    Load ethanol dataset and create data loaders.

    Args:
        data_dir: path to data/md17_ethanol_v1
        seq_length: 9 for real atoms only, 21 for padded

    Returns:
        (train_loader, val_loader, mask_np, atom_types_np, ref_positions_np)
    """
    train_set = MD17Dataset(data_dir, split='train')
    val_set = MD17Dataset(data_dir, split='val')

    # mask is (21,); atom_types is (21,) int
    mask_np = train_set.mask.numpy()     # (21,) with 1.0 for real atoms
    atom_types_np = train_set.atom_types.numpy()  # (21,) int, H=0,C=1,N=2,O=3

    # Reference positions from validation set (for evaluation)
    ref_positions_all = val_set.positions.numpy()  # (N_val, 21, 3)

    return train_set, val_set, mask_np, atom_types_np, ref_positions_all


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def save_loss_curve(losses: list, logdets: list, exp_dir: Path, step_name: str):
    """Save loss curve with log-det track."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    steps = np.arange(len(losses))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax1.plot(steps, losses, linewidth=0.8, color='steelblue', alpha=0.7)
    if len(losses) > 50:
        window = max(len(losses) // 50, 10)
        smoothed = np.convolve(losses, np.ones(window) / window, mode='valid')
        ax1.plot(np.arange(len(smoothed)) + window // 2, smoothed,
                 linewidth=2, color='darkblue', label=f'Smoothed (w={window})')
    best_step = int(np.argmin(losses))
    ax1.axvline(best_step, color='red', linestyle='--', alpha=0.5)
    ax1.annotate(f'Best: {losses[best_step]:.4f}@{best_step}',
                 xy=(best_step, losses[best_step]),
                 xytext=(best_step + len(losses) * 0.03, losses[best_step]),
                 fontsize=8, color='red')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('NLL Loss')
    ax1.set_title(f'{step_name} — Training Loss')
    ax1.grid(True, alpha=0.3)
    if len(losses) > 0 and max(losses) / (abs(min(losses)) + 1e-6) > 20:
        ax1.set_yscale('symlog')

    # Log-det/dof
    if logdets:
        ax2.plot(np.arange(len(logdets)), logdets, linewidth=0.8, color='darkorange', alpha=0.7)
        if len(logdets) > 50:
            window2 = max(len(logdets) // 50, 10)
            smoothed2 = np.convolve(logdets, np.ones(window2) / window2, mode='valid')
            ax2.plot(np.arange(len(smoothed2)) + window2 // 2, smoothed2,
                     linewidth=2, color='firebrick', label=f'Smoothed')
        ax2.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax2.set_xlabel('Step')
        ax2.set_ylabel('log_det / dof')
        ax2.set_title(f'{step_name} — log_det/dof')
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = exp_dir / 'loss_curve.png'
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return save_path


def save_pairwise_dist_comparison(
    gen_positions: np.ndarray,
    ref_positions: np.ndarray,
    mask: np.ndarray,
    exp_dir: Path,
    step_name: str,
    valid_frac: float,
):
    """Save comparison of generated vs reference pairwise distance distributions."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    gen_hist, edges = metrics_module.pairwise_distance_histogram(gen_positions, mask)
    ref_hist, _ = metrics_module.pairwise_distance_histogram(ref_positions, mask)

    bin_centers = 0.5 * (edges[:-1] + edges[1:])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.fill_between(bin_centers, ref_hist, alpha=0.4, color='orange', label='Reference (val)')
    ax.plot(bin_centers, ref_hist, color='darkorange', linewidth=1)
    ax.fill_between(bin_centers, gen_hist, alpha=0.4, color='steelblue', label=f'Generated')
    ax.plot(bin_centers, gen_hist, color='darkblue', linewidth=1)
    ax.axvline(0.8, color='red', linestyle='--', alpha=0.7, label='Min valid dist (0.8 Å)')
    ax.set_xlabel('Pairwise distance (Å)')
    ax.set_ylabel('Normalized frequency')
    ax.set_title(f'{step_name} — Pairwise Distance Distribution\nValid fraction: {valid_frac:.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = exp_dir / 'pairwise_dist.png'
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return save_path


# ---------------------------------------------------------------------------
# Core training loop
# ---------------------------------------------------------------------------

def train_step(
    step_name: str,
    cfg: dict,
    device: torch.device,
    exp_dir: Path,
    model: TarFlow1DMol,
    train_set,
    val_set,
    mask_np: np.ndarray,
    atom_types_np: np.ndarray,
    ref_positions_np: np.ndarray,
    n_real: int,
):
    """
    Unified training loop for all steps.

    Args:
        step_name: 'a', 'b', 'c', 'd', 'e', 'f'
        cfg: configuration dict
        device: torch device
        exp_dir: output directory
        model: the model to train (already on device)
        train_set, val_set: MD17Dataset instances
        mask_np: (21,) float numpy array
        atom_types_np: (21,) int numpy array
        ref_positions_np: (N_val, 21, 3) reference positions
        n_real: number of real atoms in this step's sequences
    """
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(cfg, f, indent=2)

    # Data loaders
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=cfg['batch_size'], shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True
    )
    train_iter = iter(train_loader)

    # Optimizer + scheduler
    optimizer = optim.Adam(model.parameters(), lr=cfg['lr'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['steps'])

    # W&B
    run_name = f"und_001_phase3_step_{step_name}"
    wandb.init(
        project='tnafmol',
        name=run_name,
        group='und_001',
        tags=['understanding', 'und_001', 'DIAGNOSE', 'phase3', f'step_{step_name}', 'ethanol'],
        config=cfg,
        notes=f'Phase 3 Step {step_name.upper()}: {STEP_DESCRIPTIONS[step_name]}',
        dir='/tmp',
        reinit=True,
    )
    assert wandb.run is not None, "W&B init failed"
    wandb.log({'setup_check': 1.0}, step=0)
    print(f"W&B run: {wandb.run.url}")

    # Prepare static tensors for this step
    use_padding = cfg.get('use_padding_mask', False)
    use_atom_type = cfg.get('use_atom_type_cond', False)
    use_noise = cfg.get('use_noise', False)
    noise_sigma = cfg.get('noise_sigma', 0.0)
    use_shared_scale = cfg.get('use_shared_scale', False)

    # mask for real atoms (for loss, evaluation)
    mask_tensor = torch.from_numpy(mask_np).to(device)  # (21,)
    # atom types
    at_full = torch.from_numpy(atom_types_np).long().to(device)  # (21,)
    at_real = at_full[:n_real]  # (n_real,)

    losses = []
    logdets_track = []  # track logdet/dof
    best_loss = float('inf')
    best_step = 0

    n_dof = n_real * 3  # degrees of freedom for real atoms

    model.train()
    for step in range(cfg['steps']):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        positions_full = batch['positions'].to(device)  # (B, 21, 3)
        B = positions_full.size(0)

        # Extract positions for this step
        if use_padding:
            # Steps C, D, E, F: use all 21 atoms with padding mask
            x = positions_full  # (B, 21, 3)
            pad_mask = mask_tensor.unsqueeze(0).expand(B, -1)  # (B, 21)
            atom_types_batch = at_full.unsqueeze(0).expand(B, -1)  # (B, 21) if needed
        else:
            # Steps A, B: use only 9 real atoms
            x = positions_full[:, :n_real, :]  # (B, 9, 3)
            pad_mask = None
            atom_types_batch = at_real.unsqueeze(0).expand(B, -1)  # (B, 9) if needed

        # Noise augmentation (Steps D, E, F)
        if use_noise and noise_sigma > 0:
            noise = noise_sigma * torch.randn_like(x)
            if use_padding:
                noise = noise * mask_tensor.unsqueeze(0).unsqueeze(-1)  # zero out padding
            x = x + noise

        optimizer.zero_grad()

        # Forward pass
        if use_atom_type:
            z, logdets = model(x, atom_types=atom_types_batch, padding_mask=pad_mask)
        elif use_padding:
            z, logdets = model(x, padding_mask=pad_mask)
        else:
            z, logdets = model(x)

        # Loss
        loss, info = model.get_loss(z, logdets, padding_mask=pad_mask)

        if not torch.isfinite(loss):
            print(f"  WARNING: Non-finite loss at step {step}: {loss.item()} — skipping step")
            optimizer.zero_grad()
            scheduler.step()
            # Use last finite loss for tracking
            loss_val = losses[-1] if losses else 0.0
            losses.append(loss_val)
            logdets_track.append(logdets_track[-1] if logdets_track else 0.0)
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        loss_val = loss.item()
        losses.append(loss_val)

        # Track log_det/dof
        logdet_per_dof = info['logdets_mean'] / n_dof
        logdets_track.append(logdet_per_dof)

        if loss_val < best_loss:
            best_loss = loss_val
            best_step = step
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': cfg,
                'best_loss': best_loss,
                'n_real': n_real,
                'mask_np': mask_np,
                'atom_types_np': atom_types_np,
            }, exp_dir / 'best.pt')

        if step % 100 == 0 or step == cfg['steps'] - 1:
            lr_now = scheduler.get_last_lr()[0]
            print(f"  Step {step:5d} | loss={loss_val:.4f} | logdet/dof={logdet_per_dof:.4f} | lr={lr_now:.2e}")
            wandb.log({
                'train/loss': loss_val,
                'train/nll': info['nll'],
                'train/reg': info['reg'],
                'train/logdet_per_dof': logdet_per_dof,
                'train/logdets_mean': info['logdets_mean'],
                'train/lr': lr_now,
            }, step=step)

        # Periodic checkpoint (every 1000 steps)
        if (step + 1) % 1000 == 0:
            ckpt_path = exp_dir / f'checkpoint_step{step+1}.pt'
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'config': cfg,
                'best_loss': best_loss,
            }, ckpt_path)
            # Keep only last 2
            ckpts = sorted(exp_dir.glob('checkpoint_step*.pt'),
                           key=lambda p: int(p.stem.split('step')[1]))
            for old in ckpts[:-2]:
                old.unlink()

    # Evaluation: generate 1000 samples and compute metrics
    model.eval()
    print(f"\n  Evaluating {step_name.upper()}...")

    with torch.no_grad():
        if use_padding:
            # Generate with padding: full 21-atom sequences
            pad_mask_eval = mask_tensor.unsqueeze(0).expand(1000, -1)
            at_eval = at_full.unsqueeze(0).expand(1000, -1)
            if use_atom_type:
                samples = model.sample(1000, device=device,
                                       atom_types=at_eval, padding_mask=pad_mask_eval)
            else:
                samples = model.sample(1000, device=device, padding_mask=pad_mask_eval)
        else:
            # Generate real atoms only
            if use_atom_type:
                at_eval = at_real.unsqueeze(0).expand(1000, -1)
                samples = model.sample(1000, device=device, atom_types=at_eval)
            else:
                samples = model.sample(1000, device=device)

    samples_np = samples.cpu().numpy()  # (1000, T, 3)

    # Pad generated samples to 21 atoms if needed (for metrics)
    if not use_padding:
        # samples_np is (1000, 9, 3) — need to embed in (1000, 21, 3) for metrics
        gen_padded = np.zeros((1000, 21, 3), dtype=np.float32)
        gen_padded[:, :n_real, :] = samples_np
    else:
        gen_padded = samples_np  # already (1000, 21, 3)

    # Evaluation mask always uses the real atom mask
    valid_frac, per_sample_valid = metrics_module.valid_fraction(gen_padded, mask_np)
    print(f"  Valid fraction: {valid_frac:.4f}")

    # NLL/dof = final_loss (Apple NLL is already per dimension)
    final_loss = losses[-1]
    nll_per_dof = final_loss  # Apple loss averages over dims

    # log_det/dof tracking
    avg_logdet_per_dof = np.mean(logdets_track[-100:]) if logdets_track else 0.0

    print(f"  Final loss: {final_loss:.4f}")
    print(f"  NLL/dof: {nll_per_dof:.4f}")
    print(f"  log_det/dof (last 100 steps): {avg_logdet_per_dof:.4f}")

    # Reference positions (first 1000 val samples)
    ref_sub = ref_positions_np[:1000]  # (1000, 21, 3)

    # Visualizations
    loss_path = save_loss_curve(losses, logdets_track, exp_dir, f'und_001_phase3_step_{step_name}')
    dist_path = save_pairwise_dist_comparison(
        gen_padded, ref_sub, mask_np, exp_dir,
        f'und_001_phase3_step_{step_name}',
        valid_frac,
    )

    # Save raw outputs
    raw_dir = exp_dir / 'raw'
    raw_dir.mkdir(exist_ok=True)
    np.save(raw_dir / 'generated_positions.npy', gen_padded)
    np.save(raw_dir / 'per_sample_valid.npy', per_sample_valid)
    np.save(raw_dir / 'losses.npy', np.array(losses))
    np.save(raw_dir / 'logdets_track.npy', np.array(logdets_track))

    # W&B summary
    wandb.log({
        'final/loss_curve': wandb.Image(str(loss_path)),
        'final/pairwise_dist': wandb.Image(str(dist_path)),
        'eval/valid_fraction': valid_frac,
    })
    wandb.run.summary.update({
        'best_loss': best_loss,
        'best_step': best_step,
        'final_loss': final_loss,
        'valid_fraction': valid_frac,
        'nll_per_dof': nll_per_dof,
        'logdet_per_dof': avg_logdet_per_dof,
        'n_params': cfg.get('n_params', 0),
    })

    artifact = wandb.Artifact(f'und_001_phase3_step_{step_name}_model', type='model')
    artifact.add_file(str(exp_dir / 'best.pt'))
    wandb.log_artifact(artifact)

    wandb.finish()

    results = {
        'step': step_name,
        'description': STEP_DESCRIPTIONS[step_name],
        'best_loss': best_loss,
        'best_step': best_step,
        'final_loss': final_loss,
        'valid_fraction': valid_frac,
        'nll_per_dof': nll_per_dof,
        'logdet_per_dof': avg_logdet_per_dof,
        'n_params': cfg.get('n_params', 0),
    }

    with open(exp_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n  Step {step_name.upper()} complete:")
    print(f"    final_loss={final_loss:.4f}, valid_fraction={valid_frac:.4f}")
    print(f"    logdet/dof={avg_logdet_per_dof:.4f}")

    return results


# ---------------------------------------------------------------------------
# Step builders — construct model + config for each step
# ---------------------------------------------------------------------------

def build_step_a(device, seed, project_root):
    """Step A: Raw atomic coordinates — pure Apple TarFlow1D on 9 ethanol atoms."""
    cfg = {
        'step': 'a',
        'description': STEP_DESCRIPTIONS['a'],
        'exp_id': 'und_001',
        'command': 'DIAGNOSE',
        'seed': seed,
        'device': str(device),
        'git_hash': _get_git_hash(project_root),
        # Data
        'n_real': N_REAL_ATOMS,
        'seq_length': N_REAL_ATOMS,
        'use_atom_type_cond': False,
        'use_padding_mask': False,
        'use_noise': False,
        'use_shared_scale': False,
        'use_clamp': False,
        'log_det_reg_weight': 0.0,
        # Model (Apple defaults)
        'in_channels': 3,
        'channels': 256,
        'num_blocks': 4,
        'layers_per_block': 2,
        'head_dim': 64,
        # Training
        'steps': 5000,
        'batch_size': 256,
        'lr': 5e-4,
        'lr_schedule': 'cosine',
        'grad_clip_norm': 1.0,
    }

    # Pure Apple TarFlow1D (no modifications)
    model = TarFlow1D(
        in_channels=cfg['in_channels'],
        seq_length=cfg['seq_length'],
        channels=cfg['channels'],
        num_blocks=cfg['num_blocks'],
        layers_per_block=cfg['layers_per_block'],
        head_dim=cfg['head_dim'],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    cfg['n_params'] = n_params
    print(f"Step A model: {n_params:,} parameters")

    # Wrap TarFlow1D with a compatibility shim for the shared training loop
    return StepAWrapper(model), cfg, N_REAL_ATOMS


def build_step_b(device, seed, project_root):
    """Step B: Add atom type conditioning."""
    cfg = {
        'step': 'b',
        'description': STEP_DESCRIPTIONS['b'],
        'exp_id': 'und_001',
        'command': 'DIAGNOSE',
        'seed': seed,
        'device': str(device),
        'git_hash': _get_git_hash(project_root),
        # Data
        'n_real': N_REAL_ATOMS,
        'seq_length': N_REAL_ATOMS,
        'use_atom_type_cond': True,
        'use_padding_mask': False,
        'use_noise': False,
        'use_shared_scale': False,
        'use_clamp': False,
        'log_det_reg_weight': 0.0,
        # Model
        'in_channels': 3,
        'atom_type_emb_dim': ATOM_EMB_DIM,
        'num_atom_types': NUM_ATOM_TYPES,
        'channels': 256,
        'num_blocks': 4,
        'layers_per_block': 2,
        'head_dim': 64,
        # Training
        'steps': 5000,
        'batch_size': 256,
        'lr': 5e-4,
        'lr_schedule': 'cosine',
        'grad_clip_norm': 1.0,
    }

    model = TarFlow1DMol(
        in_channels=3,
        seq_length=N_REAL_ATOMS,
        channels=cfg['channels'],
        num_blocks=cfg['num_blocks'],
        layers_per_block=cfg['layers_per_block'],
        head_dim=cfg['head_dim'],
        use_atom_type_cond=True,
        atom_type_emb_dim=ATOM_EMB_DIM,
        num_atom_types=NUM_ATOM_TYPES,
        use_padding_mask=False,
        use_shared_scale=False,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    cfg['n_params'] = n_params
    print(f"Step B model: {n_params:,} parameters")
    return model, cfg, N_REAL_ATOMS


def build_step_c(device, seed, project_root):
    """Step C: Add padding + attention masking."""
    cfg = {
        'step': 'c',
        'description': STEP_DESCRIPTIONS['c'],
        'exp_id': 'und_001',
        'command': 'DIAGNOSE',
        'seed': seed,
        'device': str(device),
        'git_hash': _get_git_hash(project_root),
        # Data
        'n_real': N_REAL_ATOMS,
        'seq_length': N_PADDED_ATOMS,  # 21
        'use_atom_type_cond': True,
        'use_padding_mask': True,
        'use_noise': False,
        'use_shared_scale': False,
        'use_clamp': False,
        'log_det_reg_weight': 0.0,
        # Model
        'in_channels': 3,
        'atom_type_emb_dim': ATOM_EMB_DIM,
        'num_atom_types': NUM_ATOM_TYPES,
        'channels': 256,
        'num_blocks': 4,
        'layers_per_block': 2,
        'head_dim': 64,
        # Training
        'steps': 5000,
        'batch_size': 256,
        'lr': 5e-4,
        'lr_schedule': 'cosine',
        'grad_clip_norm': 1.0,
    }

    model = TarFlow1DMol(
        in_channels=3,
        seq_length=N_PADDED_ATOMS,
        channels=cfg['channels'],
        num_blocks=cfg['num_blocks'],
        layers_per_block=cfg['layers_per_block'],
        head_dim=cfg['head_dim'],
        use_atom_type_cond=True,
        atom_type_emb_dim=ATOM_EMB_DIM,
        num_atom_types=NUM_ATOM_TYPES,
        use_padding_mask=True,
        use_shared_scale=False,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    cfg['n_params'] = n_params
    print(f"Step C model: {n_params:,} parameters")
    return model, cfg, N_REAL_ATOMS


def build_step_d(device, seed, project_root):
    """Step D: Add Gaussian noise augmentation."""
    cfg = {
        'step': 'd',
        'description': STEP_DESCRIPTIONS['d'],
        'exp_id': 'und_001',
        'command': 'DIAGNOSE',
        'seed': seed,
        'device': str(device),
        'git_hash': _get_git_hash(project_root),
        # Data
        'n_real': N_REAL_ATOMS,
        'seq_length': N_PADDED_ATOMS,
        'use_atom_type_cond': True,
        'use_padding_mask': True,
        'use_noise': True,
        'noise_sigma': 0.05,
        'use_shared_scale': False,
        'use_clamp': False,
        'log_det_reg_weight': 0.0,
        # Model
        'in_channels': 3,
        'atom_type_emb_dim': ATOM_EMB_DIM,
        'num_atom_types': NUM_ATOM_TYPES,
        'channels': 256,
        'num_blocks': 4,
        'layers_per_block': 2,
        'head_dim': 64,
        # Training
        'steps': 5000,
        'batch_size': 256,
        'lr': 5e-4,
        'lr_schedule': 'cosine',
        'grad_clip_norm': 1.0,
    }

    model = TarFlow1DMol(
        in_channels=3,
        seq_length=N_PADDED_ATOMS,
        channels=cfg['channels'],
        num_blocks=cfg['num_blocks'],
        layers_per_block=cfg['layers_per_block'],
        head_dim=cfg['head_dim'],
        use_atom_type_cond=True,
        atom_type_emb_dim=ATOM_EMB_DIM,
        num_atom_types=NUM_ATOM_TYPES,
        use_padding_mask=True,
        use_shared_scale=False,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    cfg['n_params'] = n_params
    print(f"Step D model: {n_params:,} parameters")
    return model, cfg, N_REAL_ATOMS


def build_step_e(device, seed, project_root):
    """Step E: Switch to shared scale — KEY TEST."""
    cfg = {
        'step': 'e',
        'description': STEP_DESCRIPTIONS['e'],
        'exp_id': 'und_001',
        'command': 'DIAGNOSE',
        'seed': seed,
        'device': str(device),
        'git_hash': _get_git_hash(project_root),
        # Data
        'n_real': N_REAL_ATOMS,
        'seq_length': N_PADDED_ATOMS,
        'use_atom_type_cond': True,
        'use_padding_mask': True,
        'use_noise': True,
        'noise_sigma': 0.05,
        'use_shared_scale': True,   # KEY CHANGE
        'use_clamp': False,
        'log_det_reg_weight': 0.0,
        # Model
        'in_channels': 3,
        'atom_type_emb_dim': ATOM_EMB_DIM,
        'num_atom_types': NUM_ATOM_TYPES,
        'channels': 256,
        'num_blocks': 4,
        'layers_per_block': 2,
        'head_dim': 64,
        # Training
        'steps': 5000,
        'batch_size': 256,
        'lr': 5e-4,
        'lr_schedule': 'cosine',
        'grad_clip_norm': 1.0,
    }

    model = TarFlow1DMol(
        in_channels=3,
        seq_length=N_PADDED_ATOMS,
        channels=cfg['channels'],
        num_blocks=cfg['num_blocks'],
        layers_per_block=cfg['layers_per_block'],
        head_dim=cfg['head_dim'],
        use_atom_type_cond=True,
        atom_type_emb_dim=ATOM_EMB_DIM,
        num_atom_types=NUM_ATOM_TYPES,
        use_padding_mask=True,
        use_shared_scale=True,
        use_clamp=False,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    cfg['n_params'] = n_params
    print(f"Step E model: {n_params:,} parameters")
    return model, cfg, N_REAL_ATOMS


def build_step_f(device, seed, project_root):
    """Step F: Add stabilization (asymmetric clamp + log-det reg)."""
    cfg = {
        'step': 'f',
        'description': STEP_DESCRIPTIONS['f'],
        'exp_id': 'und_001',
        'command': 'DIAGNOSE',
        'seed': seed,
        'device': str(device),
        'git_hash': _get_git_hash(project_root),
        # Data
        'n_real': N_REAL_ATOMS,
        'seq_length': N_PADDED_ATOMS,
        'use_atom_type_cond': True,
        'use_padding_mask': True,
        'use_noise': True,
        'noise_sigma': 0.05,
        'use_shared_scale': True,
        'use_clamp': True,          # KEY CHANGE from Step E
        'alpha_pos': 0.1,
        'alpha_neg': 2.0,
        'log_det_reg_weight': 0.01,  # KEY CHANGE from Step E
        # Model
        'in_channels': 3,
        'atom_type_emb_dim': ATOM_EMB_DIM,
        'num_atom_types': NUM_ATOM_TYPES,
        'channels': 256,
        'num_blocks': 4,
        'layers_per_block': 2,
        'head_dim': 64,
        # Training
        'steps': 5000,
        'batch_size': 256,
        'lr': 5e-4,
        'lr_schedule': 'cosine',
        'grad_clip_norm': 1.0,
    }

    model = TarFlow1DMol(
        in_channels=3,
        seq_length=N_PADDED_ATOMS,
        channels=cfg['channels'],
        num_blocks=cfg['num_blocks'],
        layers_per_block=cfg['layers_per_block'],
        head_dim=cfg['head_dim'],
        use_atom_type_cond=True,
        atom_type_emb_dim=ATOM_EMB_DIM,
        num_atom_types=NUM_ATOM_TYPES,
        use_padding_mask=True,
        use_shared_scale=True,
        use_clamp=True,
        alpha_pos=0.1,
        alpha_neg=2.0,
        log_det_reg_weight=0.01,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    cfg['n_params'] = n_params
    print(f"Step F model: {n_params:,} parameters")
    return model, cfg, N_REAL_ATOMS


# ---------------------------------------------------------------------------
# StepAWrapper — wraps TarFlow1D to match TarFlow1DMol interface
# ---------------------------------------------------------------------------

class StepAWrapper(nn.Module):
    """
    Thin wrapper around TarFlow1D to present the same interface as TarFlow1DMol
    for use in the unified training loop.
    """

    def __init__(self, base: TarFlow1D):
        super().__init__()
        self.base = base

    def forward(self, x, atom_types=None, padding_mask=None):
        z, logdets = self.base(x)
        return z, logdets

    def get_loss(self, z, logdets, padding_mask=None):
        loss = self.base.get_loss(z, logdets)
        info = {
            'nll': loss.item(),
            'reg': 0.0,
            'logdets_mean': logdets.mean().item(),
        }
        return loss, info

    @torch.no_grad()
    def sample(self, n, device, atom_types=None, padding_mask=None, temp=1.0):
        return self.base.sample(n, device=device, temp=temp)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _get_git_hash(project_root):
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=project_root
        ).decode().strip()
    except Exception:
        return 'unknown'


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

STEP_BUILDERS = {
    'a': build_step_a,
    'b': build_step_b,
    'c': build_step_c,
    'd': build_step_d,
    'e': build_step_e,
    'f': build_step_f,
}


def parse_args():
    parser = argparse.ArgumentParser(description='Phase 3 Adaptation Ladder')
    parser.add_argument('--step', type=str, required=True,
                        choices=['a', 'b', 'c', 'd', 'e', 'f', 'all'],
                        help='Step to run (a-f) or "all" to run all')
    parser.add_argument('--gpu', type=int, default=0, help='Logical GPU index')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(args.gpu)}")
        print(f"Memory: {torch.cuda.get_device_properties(args.gpu).total_memory / 1e9:.1f} GB")

    project_root = str(Path(__file__).parent.parent)
    data_dir = str(Path(project_root) / ETHANOL_DATA_DIR)

    # Load ethanol data (full 21-atom version — steps that need < 21 atoms will slice)
    print(f"\nLoading ethanol data from {data_dir}...")
    train_set, val_set, mask_np, atom_types_np, ref_positions_np = load_ethanol(data_dir)
    print(f"  Train: {len(train_set)}, Val: {len(val_set)}")
    print(f"  Mask n_real: {int(mask_np.sum())}")

    steps_to_run = list('abcdef') if args.step == 'all' else [args.step]
    all_results = []

    for step_name in steps_to_run:
        print(f"\n{'='*60}")
        print(f"STEP {step_name.upper()}: {STEP_DESCRIPTIONS[step_name]}")
        print(f"{'='*60}")

        set_seed(args.seed)  # reset seed for each step (fresh init)

        builder = STEP_BUILDERS[step_name]
        model, cfg, n_real = builder(device, args.seed, project_root)
        cfg['device'] = str(device)

        exp_dir = Path(project_root) / RESULTS_BASE / STEP_DIRS[step_name]

        results = train_step(
            step_name=step_name,
            cfg=cfg,
            device=device,
            exp_dir=exp_dir,
            model=model,
            train_set=train_set,
            val_set=val_set,
            mask_np=mask_np,
            atom_types_np=atom_types_np,
            ref_positions_np=ref_positions_np,
            n_real=n_real,
        )
        all_results.append(results)

        # Print running summary
        print(f"\n  ---- Step {step_name.upper()} Summary ----")
        print(f"  Valid fraction: {results['valid_fraction']:.4f}")
        print(f"  Final loss: {results['final_loss']:.4f}")
        print(f"  log_det/dof: {results['logdet_per_dof']:.4f}")

    # Print final summary table
    if len(all_results) > 1:
        print(f"\n{'='*80}")
        print(f"PHASE 3 ADAPTATION LADDER — SUMMARY")
        print(f"{'='*80}")
        print(f"{'Step':<6} {'Description':<40} {'Loss':>8} {'Valid%':>8} {'logd/dof':>10}")
        print(f"{'-'*6} {'-'*40} {'-'*8} {'-'*8} {'-'*10}")
        for r in all_results:
            desc = r['description'][:38]
            print(f"  {r['step'].upper():<4} {desc:<40} {r['final_loss']:>8.4f} "
                  f"{r['valid_fraction']*100:>7.1f}% {r['logdet_per_dof']:>10.4f}")

    return all_results


if __name__ == '__main__':
    main()
