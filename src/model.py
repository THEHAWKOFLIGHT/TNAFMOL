"""
TNAFMOL Model — TarFlow: Transformer Autoregressive Normalizing Flow

Architecture:
- Stack of L autoregressive transformer blocks (default L=8)
- Each block applies causal masked self-attention over atoms, predicts per-atom affine
  parameters (shift_i: R^3, log_scale_i: scalar), transforms coordinates.
- Autoregressive ordering is over ATOMS. Within each atom, all 3 coordinates are
  transformed simultaneously using the same affine parameters predicted from
  previous atoms.
- Even blocks: forward ordering (atom 0 → atom N-1)
- Odd blocks: reverse ordering (atom N-1 → atom 0)
- Atom type conditioning: learned embedding concatenated at every layer.
- Base distribution: isotropic Gaussian N(0, I) over all real atom coordinates.

Implementation detail:
- A learned SOS (start-of-sequence) token is prepended to the atom sequence.
- Atom i attends to [SOS, atom_0, ..., atom_{i-1}] in forward order.
- This ensures atom 0 always has a valid context (the SOS token), avoiding NaN softmax.
- The SOS token output is discarded; only atom outputs are used.

Training:
- MLE: minimize NLL = -sum_i(3 * log_scale_i * mask_i) - log p_z(z)
  where z = forward(x), p_z is standard Gaussian.
  The log-det contribution from each atom is 3 * log_scale_i (since 3 coords scaled by same factor).
- Optional log-det regularization: penalizes large log_det_per_dof values to prevent exploitation.

Sampling:
- z ~ N(0, I) for real atoms (padded atoms stay at 0)
- Apply inverse transforms in reverse block order
  Inverse per atom: x_i = (y_i - shift_i) / exp(log_scale_i)
  Applied autoregressively.

hyp_003 changes:
- Replace tanh*log_scale_max clamping with asymmetric arctan clamping (Andrade et al. 2024)
  alpha_pos bounds expansion, alpha_neg allows contraction — prevents log_det exploitation
- Add log_det_reg_weight penalty on log_det_per_dof^2 to explicitly penalize exploitation
- Add EMAModel class for heuristics angle
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# =============================================================================
# Asymmetric soft clamping (Andrade et al. 2024)
# =============================================================================

def _asymmetric_clamp(s: torch.Tensor, alpha_pos: float = 0.1, alpha_neg: float = 2.0) -> torch.Tensor:
    """Asymmetric soft clamping via arctan (Andrade et al. 2024).

    For s >= 0: c(s) = (2/pi) * alpha_pos * arctan(s / alpha_pos)
      -> bounds expansion to exp(alpha_pos) ~1.105x per layer
    For s < 0: c(s) = (2/pi) * alpha_neg * arctan(s / alpha_neg)
      -> allows contraction up to exp(-alpha_neg) per layer

    Sanity checks:
    - At s=0: c(0) = 0 (continuous, no discontinuity). CHECK.
    - As s -> +inf: c(s) -> alpha_pos (hard upper bound on expansion). CHECK.
    - As s -> -inf: c(s) -> -alpha_neg (hard lower bound, allows contraction). CHECK.
    - Derivative at 0: (2/pi) * 1 for both sides = continuous derivative. CHECK.
    - Symmetric case (alpha_pos = alpha_neg = alpha): c(s) = (2/pi)*alpha*arctan(s/alpha),
      standard symmetric soft clamp used in Andrade et al. CHECK.

    Args:
        s: (B, N, 1) or any shape — raw log_scale predictions
        alpha_pos: soft bound for positive s (expansion limit)
        alpha_neg: soft bound for negative s (contraction limit)

    Returns:
        clamped s, same shape, bounded in (-alpha_neg, alpha_pos)
    """
    pos_mask = (s >= 0).float()
    neg_mask = 1.0 - pos_mask
    clamped = (
        pos_mask * (2.0 / math.pi) * alpha_pos * torch.atan(s / alpha_pos)
        + neg_mask * (2.0 / math.pi) * alpha_neg * torch.atan(s / alpha_neg)
    )
    return clamped


# =============================================================================
# ActNorm (kept for backward compatibility, not used in hyp_003)
# =============================================================================

class ActNorm(nn.Module):
    """Activation Normalization for normalizing flows.

    Per-atom affine normalization with data-dependent initialization.
    Normalizes each atom's 3D position to zero mean and unit std using
    learned shift and log_scale parameters. Initialized on the first batch.

    Reference: Kingma & Dhariwal, "Glow: Generative flow with invertible 1x1
    convolutions", NeurIPS 2018.

    Args:
        max_atoms: maximum number of atoms (padded dimension size)
    """

    def __init__(self, max_atoms: int = 21):
        super().__init__()
        self.max_atoms = max_atoms
        # Learned per-atom shift and log_scale: shape (1, max_atoms, 3)
        # shift: subtract the learned mean
        # log_scale: divide by exp(log_scale) = learned std
        self.shift = nn.Parameter(torch.zeros(1, max_atoms, 3))
        self.log_scale = nn.Parameter(torch.zeros(1, max_atoms, 3))
        self.initialized = False

    @torch.no_grad()
    def initialize(self, positions: torch.Tensor, atom_mask: torch.Tensor):
        """Data-dependent initialization: set shift and log_scale from first batch.

        After init, the output y = (x - shift) * exp(-log_scale) ~ N(0,1) per atom.

        Args:
            positions: (B, N, 3) input positions
            atom_mask: (B, N) or (N,) 1=real, 0=padding
        """
        B, N, _ = positions.shape

        # Broadcast mask to (B, N, 1)
        if atom_mask.dim() == 1:
            mask = atom_mask.unsqueeze(0).expand(B, -1)  # (B, N)
        else:
            mask = atom_mask  # (B, N)
        mask_expanded = mask.unsqueeze(-1).float()  # (B, N, 1)

        # Count valid (real) entries per atom slot: sum over batch dimension
        # valid_count[j] = number of batches where atom j is real
        valid_count = mask_expanded.sum(dim=0, keepdim=True)  # (1, N, 1)
        valid_count = valid_count.clamp(min=1.0)

        # Per-atom mean (average over batch, only counting real atoms)
        mean = (positions * mask_expanded).sum(dim=0, keepdim=True) / valid_count  # (1, N, 3)

        # Per-atom std (average over batch, only counting real atoms)
        diff = (positions - mean) * mask_expanded  # (B, N, 3)
        var = (diff ** 2).sum(dim=0, keepdim=True) / valid_count  # (1, N, 3)
        std = (var + 1e-6).sqrt()  # (1, N, 3)

        # Set parameters: forward transform is y = (x - shift) * exp(-log_scale)
        # So to normalize: shift = mean, exp(log_scale) = std -> log_scale = log(std)
        # Check: y = (x - mean) * exp(-log(std)) = (x - mean) / std ~ N(0,1)
        self.shift.data.copy_(mean)
        self.log_scale.data.copy_(std.log())
        self.initialized = True

    def forward(
        self,
        positions: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ActNorm forward: y = (x - shift) * exp(-log_scale).

        Args:
            positions: (B, N, 3) input
            atom_mask: (B, N) or (N,) 1=real, 0=padding

        Returns:
            y: (B, N, 3) normalized positions
            log_det: (B,) log-determinant contribution
        """
        if not self.initialized:
            # Lazy init on first forward call
            self.initialize(positions, atom_mask)

        if atom_mask.dim() == 1:
            mask = atom_mask.unsqueeze(0).expand(positions.shape[0], -1)
        else:
            mask = atom_mask

        # Normalize: y = (x - shift) * exp(-log_scale)
        y = (positions - self.shift) * (-self.log_scale).exp()

        # Zero out padding
        y = y * mask.unsqueeze(-1)

        # Log-det: each real atom contributes -3 * sum(log_scale) per coordinate
        # log_det per sample = sum_real_atoms(-3 * log_scale_per_coord)
        # = sum_real_atoms(-log_scale_x - log_scale_y - log_scale_z)
        # But log_scale is (1, N, 3), so sum over coords:
        neg_log_scale_sum = -self.log_scale.sum(dim=-1)  # (1, N)
        log_det = (neg_log_scale_sum * mask).sum(dim=-1)  # (B,)

        return y, log_det

    def inverse(
        self,
        y: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> torch.Tensor:
        """ActNorm inverse: x = y * exp(log_scale) + shift.

        Args:
            y: (B, N, 3) normalized positions
            atom_mask: (B, N) or (N,) 1=real, 0=padding

        Returns:
            x: (B, N, 3) unnormalized positions
        """
        if atom_mask.dim() == 1:
            mask = atom_mask.unsqueeze(0).expand(y.shape[0], -1)
        else:
            mask = atom_mask

        x = y * self.log_scale.exp() + self.shift
        return x * mask.unsqueeze(-1)


# =============================================================================
# TarFlow Block
# =============================================================================

class TarFlowBlock(nn.Module):
    """Single autoregressive transformer block.

    Uses a SOS (start-of-sequence) token prepended to the atom sequence.
    Atom i attends to [SOS, atom_0, ..., atom_{i-1}], guaranteeing a valid
    attention context even for atom 0.

    Args:
        d_model: transformer hidden dimension
        n_heads: number of attention heads
        ffn_mult: FFN hidden dim = ffn_mult * d_model
        in_features: dimension of input per atom (pos_dim + atom_type_emb_dim)
        reverse: if True, use reverse autoregressive ordering
        dropout: dropout rate
        alpha_pos: soft clamp bound for positive log_scale (expansion limit)
        alpha_neg: soft clamp bound for negative log_scale (contraction limit)
        shift_only: if True, use volume-preserving (shift-only) flow
        log_scale_max: DEPRECATED — kept for backward compat, ignored if alpha_pos/alpha_neg provided
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_mult: int = 4,
        in_features: int = 19,  # 3 (pos) + 16 (atom type emb)
        reverse: bool = False,
        dropout: float = 0.1,
        alpha_pos: float = 0.1,
        alpha_neg: float = 2.0,
        shift_only: bool = False,
        log_scale_max: float = 0.5,  # DEPRECATED — ignored, kept for compat
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.reverse = reverse
        self.alpha_pos = alpha_pos
        self.alpha_neg = alpha_neg
        self.shift_only = shift_only

        # Learnable SOS token (provides context for the first atom)
        self.sos = nn.Parameter(torch.randn(1, 1, d_model) * 0.01)

        # Input projection: in_features -> d_model
        self.input_proj = nn.Linear(in_features, d_model)

        # Multi-head attention
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn_dropout = nn.Dropout(dropout)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(d_model)

        # Output head:
        #   shift_only=True: d_model -> shift(3) — volume-preserving (no scale)
        #   shift_only=False: d_model -> shift(3) + log_scale(1) per atom
        out_dim = 3 if shift_only else 4
        self.out_proj = nn.Linear(d_model, out_dim)

        # Initialize output projection near zero for stable start
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def _build_causal_mask(self, n_atoms: int, device) -> torch.Tensor:
        """Build causal attention mask for n_atoms + 1 (SOS) positions.

        For forward order: atom i (at position i+1 in the sequence with SOS at 0)
        attends to [SOS, atom_0, ..., atom_{i-1}] = positions [0, 1, ..., i].
        So in the full (N+1) x (N+1) attention matrix:
          row (i+1) can attend to columns [0, 1, ..., i] (self-inclusive in the SOS+atom frame).

        For reverse order: atom i attends to [SOS, atom_{i+1}, ..., atom_{N-1}].
        We reorder the atom sequence, so atom at index j in the reversed list sees positions before it.

        Returns:
            attn_bias: (N+1, N+1) additive bias — 0 for allowed, -inf for masked
        """
        N1 = n_atoms + 1  # +1 for SOS
        # Create allowed mask: (N1, N1) bool, True = allowed
        allowed = torch.zeros(N1, N1, dtype=torch.bool, device=device)

        if not self.reverse:
            # Forward: position i (0-indexed, 0=SOS) attends to 0..i
            for i in range(N1):
                allowed[i, :i + 1] = True
        else:
            # Reverse: SOS at position 0 always allowed as context.
            # We want atom_{N-1} to be "first" and attend only to SOS,
            # atom_{N-2} to attend to SOS + atom_{N-1}, etc.
            # We achieve this by: in forward form, atom at seq position i+1 (original atom i in the reversed sequence)
            # attends to [SOS, seq_position_1, ..., seq_position_i].
            # The atoms are passed in reverse order externally (positions reversed).
            # So we just use the same self-inclusive causal mask.
            for i in range(N1):
                allowed[i, :i + 1] = True

        attn_bias = torch.zeros(N1, N1, device=device)
        attn_bias[~allowed] = float("-inf")
        return attn_bias

    def _build_padding_mask(self, atom_mask: torch.Tensor, n_atoms: int) -> torch.Tensor:
        """Build key_padding_mask for SOS + atoms.

        SOS token is never masked. Real atoms are not masked. Padding atoms are masked.

        Args:
            atom_mask: (B, N) 1=real, 0=padding
        Returns:
            kpm: (B, N+1) bool — True where position should be ignored as key
        """
        B = atom_mask.shape[0]
        # SOS: not masked (False)
        sos_mask = torch.zeros(B, 1, dtype=torch.bool, device=atom_mask.device)
        # Atoms: True where padding
        atom_kpm = (atom_mask < 0.5)  # (B, N) True=padding
        kpm = torch.cat([sos_mask, atom_kpm], dim=1)  # (B, N+1)
        return kpm

    def _get_context_features(
        self,
        positions: torch.Tensor,
        atom_type_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Build per-atom feature vector.

        Args:
            positions: (B, N, 3) — in the correct ordering (reversed if reverse=True)
            atom_type_emb: (B, N, emb_dim)

        Returns:
            h: (B, N+1, d_model) — SOS prepended
        """
        B, N, _ = positions.shape
        # Concatenate position and atom type embedding
        features = torch.cat([positions, atom_type_emb], dim=-1)  # (B, N, 3+emb_dim)
        # Project to d_model
        h_atoms = self.input_proj(features)  # (B, N, d_model)
        # Prepend SOS
        sos = self.sos.expand(B, -1, -1)  # (B, 1, d_model)
        h = torch.cat([sos, h_atoms], dim=1)  # (B, N+1, d_model)
        return h

    def _run_transformer(
        self,
        positions: torch.Tensor,
        atom_type_emb: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run transformer and return per-atom output (without SOS).

        Args:
            positions: (B, N, 3) in causal ordering
            atom_type_emb: (B, N, emb_dim) in causal ordering
            atom_mask: (B, N) in causal ordering

        Returns:
            atom_out: (B, N, d_model) — transformer output per atom
        """
        B, N, _ = positions.shape
        device = positions.device

        h = self._get_context_features(positions, atom_type_emb)  # (B, N+1, d_model)

        # Build causal mask: (N+1, N+1) additive bias
        causal_bias = self._build_causal_mask(N, device)  # (N+1, N+1)

        # Build padding key mask: (B, N+1) True=ignore
        kpm = self._build_padding_mask(atom_mask, N)  # (B, N+1) bool

        # Combine into a single float additive mask to avoid deprecation warning
        # combined: (B, N+1, N+1) — additive bias per batch element
        # kpm[:, j] = True means column j (key j) is padding — add -inf to all rows
        combined_mask = causal_bias.unsqueeze(0).expand(B, -1, -1).clone()  # (B, N+1, N+1)
        padding_rows = kpm.unsqueeze(1).float() * -1e9  # (B, 1, N+1) * -1e9
        combined_mask = combined_mask + padding_rows  # broadcast: (B, N+1, N+1)

        # Expand for all heads: (B*n_heads, N+1, N+1)
        combined_mask = combined_mask.unsqueeze(1).expand(
            -1, self.n_heads, -1, -1
        ).reshape(B * self.n_heads, N + 1, N + 1)

        # Self-attention — single float mask, no key_padding_mask
        h_attn, _ = self.attn(h, h, h, attn_mask=combined_mask)
        h = self.attn_norm(h + self.attn_dropout(h_attn))

        # FFN
        h = self.ffn_norm(h + self.ffn(h))

        # Drop SOS output, return only atom outputs
        atom_out = h[:, 1:, :]  # (B, N, d_model)
        return atom_out

    def forward(
        self,
        positions: torch.Tensor,
        atom_type_emb: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: transform positions and compute log-det.

        Args:
            positions: (batch, max_atoms, 3) input positions
            atom_type_emb: (batch, max_atoms, emb_dim) atom type embeddings
            atom_mask: (batch, max_atoms) 1=real, 0=padding

        Returns:
            y: (batch, max_atoms, 3) transformed positions
            log_det: (batch,) sum of 3 * log_scale values over real atoms
        """
        B, N, _ = positions.shape

        if self.reverse:
            # Flip atom ordering for reverse autoregressive direction
            # Flip along atom dimension
            pos_ordered = positions.flip(1)
            emb_ordered = atom_type_emb.flip(1)
            mask_ordered = atom_mask.flip(1)
        else:
            pos_ordered = positions
            emb_ordered = atom_type_emb
            mask_ordered = atom_mask

        # Run transformer to get per-atom context
        atom_out = self._run_transformer(pos_ordered, emb_ordered, mask_ordered)  # (B, N, d_model)

        if self.reverse:
            # Flip back to original ordering
            atom_out = atom_out.flip(1)

        # Predict affine params from context
        params = self.out_proj(atom_out)   # (B, N, 3 or 4)
        shift = params[..., :3]            # (B, N, 3)

        if self.shift_only:
            # Volume-preserving: y_i = x_i + shift_i, log_det = 0
            y = positions + shift
            log_det = torch.zeros(positions.shape[0], device=positions.device)
        else:
            log_scale = params[..., 3:4]       # (B, N, 1)

            # Asymmetric soft clamp via arctan (Andrade et al. 2024)
            # Bounds expansion to exp(alpha_pos) per layer, allows contraction to exp(-alpha_neg)
            log_scale = _asymmetric_clamp(log_scale, self.alpha_pos, self.alpha_neg)

            # Apply affine transform: y_i = exp(log_scale_i) * x_i + shift_i
            scale = log_scale.exp()            # (B, N, 1)
            y = scale * positions + shift      # (B, N, 3)

            # Log-determinant: each real atom contributes 3 * log_scale_i
            log_scale_sq = log_scale.squeeze(-1)   # (B, N)
            log_det = (3.0 * log_scale_sq * atom_mask).sum(dim=-1)  # (B,)

        # Zero out padding positions
        y = y * atom_mask.unsqueeze(-1)

        return y, log_det

    def inverse(
        self,
        y: torch.Tensor,
        atom_type_emb: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Inverse pass: recover x from y autoregressively.

        Since shift and log_scale for atom i depend on x_{<i}, we must
        recover atoms one at a time in the causal order.

        Args:
            y: (batch, max_atoms, 3) transformed positions
            atom_type_emb: (batch, max_atoms, emb_dim)
            atom_mask: (batch, max_atoms)

        Returns:
            x: (batch, max_atoms, 3) recovered input positions
        """
        B, N, _ = y.shape
        device = y.device

        if self.reverse:
            # Causal order in reverse direction: process from atom N-1 down to 0
            order = list(range(N - 1, -1, -1))
            # Working in reversed space
            y_work = y.flip(1)
            emb_work = atom_type_emb.flip(1)
            mask_work = atom_mask.flip(1)
        else:
            order = list(range(N))
            y_work = y
            emb_work = atom_type_emb
            mask_work = atom_mask

        x_work = torch.zeros_like(y_work)

        for step in range(N):
            # Run transformer on current x_work (partially recovered)
            atom_out = self._run_transformer(x_work, emb_work, mask_work)  # (B, N, d_model)

            params = self.out_proj(atom_out)  # (B, N, 3 or 4)
            shift = params[..., :3]           # (B, N, 3)
            shift_step = shift[:, step, :]    # (B, 3)

            if self.shift_only:
                # Volume-preserving inverse: x_i = y_i - shift_i
                x_work[:, step, :] = y_work[:, step, :] - shift_step
            else:
                log_scale = params[..., 3:4]      # (B, N, 1)
                log_scale = _asymmetric_clamp(log_scale, self.alpha_pos, self.alpha_neg)
                scale = log_scale.exp()
                scale_step = scale[:, step, :]    # (B, 1)
                # Recover atom at position `step` in causal ordering
                x_work[:, step, :] = (y_work[:, step, :] - shift_step) / scale_step

        if self.reverse:
            x_work = x_work.flip(1)

        return x_work * atom_mask.unsqueeze(-1)


# =============================================================================
# TarFlow Model
# =============================================================================

class TarFlow(nn.Module):
    """TarFlow: Transformer Autoregressive Normalizing Flow for molecules.

    A stack of L TarFlowBlocks with alternating forward/reverse ordering.
    Trained with maximum likelihood (negative log-likelihood).

    Args:
        n_blocks: number of transformer blocks (default 8)
        d_model: transformer hidden dimension (default 128)
        n_heads: number of attention heads (default 4)
        ffn_mult: FFN expansion factor (default 4)
        atom_type_emb_dim: dimension of atom type embedding (default 16)
        n_atom_types: number of distinct atom types (default 4: H, C, N, O)
        max_atoms: max number of atoms (default 21 for MD17)
        dropout: dropout rate (default 0.1)
        alpha_pos: soft clamp bound for positive log_scale — bounds expansion (default 0.1)
        alpha_neg: soft clamp bound for negative log_scale — allows contraction (default 2.0)
        shift_only: if True, use volume-preserving flow (default False)
        use_actnorm: if True, add ActNorm after each block (not used in hyp_003)
        log_scale_max: DEPRECATED — kept for backward compat, ignored
    """

    def __init__(
        self,
        n_blocks: int = 8,
        d_model: int = 128,
        n_heads: int = 4,
        ffn_mult: int = 4,
        atom_type_emb_dim: int = 16,
        n_atom_types: int = 4,
        max_atoms: int = 21,
        dropout: float = 0.1,
        alpha_pos: float = 0.1,
        alpha_neg: float = 2.0,
        shift_only: bool = False,
        use_actnorm: bool = False,
        log_scale_max: float = 0.5,  # DEPRECATED — kept for backward compat
    ):
        super().__init__()
        self.n_blocks = n_blocks
        self.d_model = d_model
        self.max_atoms = max_atoms
        self.atom_type_emb_dim = atom_type_emb_dim
        self.alpha_pos = alpha_pos
        self.alpha_neg = alpha_neg
        self.shift_only = shift_only
        self.use_actnorm = use_actnorm

        # Atom type embedding (shared across all blocks)
        self.atom_type_emb = nn.Embedding(n_atom_types, atom_type_emb_dim)

        # in_features = 3 (positions) + atom_type_emb_dim
        in_features = 3 + atom_type_emb_dim

        # Stack of blocks with alternating direction
        self.blocks = nn.ModuleList([
            TarFlowBlock(
                d_model=d_model,
                n_heads=n_heads,
                ffn_mult=ffn_mult,
                in_features=in_features,
                reverse=(i % 2 == 1),  # even=forward, odd=reverse
                dropout=dropout,
                alpha_pos=alpha_pos,
                alpha_neg=alpha_neg,
                shift_only=shift_only,
            )
            for i in range(n_blocks)
        ])

        # ActNorm layers: one per block (applied after each block)
        # Only used when use_actnorm=True
        if use_actnorm:
            self.actnorm_layers = nn.ModuleList([
                ActNorm(max_atoms=max_atoms)
                for _ in range(n_blocks)
            ])
        else:
            self.actnorm_layers = None

    def _prepare_inputs(
        self,
        positions: torch.Tensor,
        atom_types: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Ensure all inputs are (B, N, ...) shaped.

        Returns:
            positions: (B, N, 3)
            atom_type_emb: (B, N, emb_dim)
            atom_mask: (B, N)
        """
        B = positions.shape[0]

        if atom_mask.dim() == 1:
            atom_mask = atom_mask.unsqueeze(0).expand(B, -1)
        if atom_types.dim() == 1:
            atom_types = atom_types.unsqueeze(0).expand(B, -1)

        atom_type_emb = self.atom_type_emb(atom_types)  # (B, N, emb_dim)
        return positions, atom_type_emb, atom_mask

    def forward(
        self,
        positions: torch.Tensor,
        atom_types: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: data -> latent space.

        Args:
            positions: (batch, max_atoms, 3) input positions
            atom_types: (max_atoms,) or (batch, max_atoms) atom type indices
            atom_mask: (max_atoms,) or (batch, max_atoms) 1=real, 0=padding

        Returns:
            z: (batch, max_atoms, 3) latent positions
            total_log_det: (batch,) total log-determinant across all blocks
        """
        B = positions.shape[0]
        device = positions.device

        positions, atom_type_emb, atom_mask = self._prepare_inputs(
            positions, atom_types, atom_mask
        )

        z = positions
        total_log_det = torch.zeros(B, device=device)

        if self.use_actnorm:
            for block, actnorm in zip(self.blocks, self.actnorm_layers):
                z, log_det = block(z, atom_type_emb, atom_mask)
                total_log_det = total_log_det + log_det
                z, an_log_det = actnorm(z, atom_mask)
                total_log_det = total_log_det + an_log_det
        else:
            for block in self.blocks:
                z, log_det = block(z, atom_type_emb, atom_mask)
                total_log_det = total_log_det + log_det

        return z, total_log_det

    def nll_loss(
        self,
        positions: torch.Tensor,
        atom_types: torch.Tensor,
        atom_mask: torch.Tensor,
        log_det_reg_weight: float = 0.0,
    ) -> Tuple[torch.Tensor, dict]:
        """Compute negative log-likelihood loss with optional log-det regularization.

        NLL = -log p(x) = -[log p_z(z) + total_log_det]
        log p_z(z) = -0.5 * sum_{real i} ||z_i||^2 - 0.5 * n_real * 3 * log(2*pi)

        Optional log-det regularization (Andrade et al. 2024, hyp_003):
        log_det_per_dof = total_log_det / (n_real * 3)
        log_det_penalty = (log_det_per_dof^2).mean()
        total_loss = nll_per_dof.mean() + log_det_reg_weight * log_det_penalty

        This penalizes large log_det values, preventing the model from exploiting
        scale DOFs to maximize log_det without learning the data distribution.

        Normalized by n_real * 3 (total degrees of freedom) for stable training.

        Sanity checks for log_det_penalty:
        - At log_det_per_dof=0: penalty=0. CHECK (no penalty when log_det is neutral).
        - As |log_det_per_dof| grows: penalty grows quadratically. CHECK (soft quadratic barrier).
        - Backward compat (log_det_reg_weight=0.0): penalty=0, no change. CHECK.

        Args:
            positions: (batch, max_atoms, 3)
            atom_types: (max_atoms,) or (batch, max_atoms)
            atom_mask: (max_atoms,) or (batch, max_atoms)
            log_det_reg_weight: weight for log-det regularization penalty (default 0.0 = off)

        Returns:
            loss: scalar — mean NLL per degree of freedom + optional log_det penalty
            info: dict with diagnostics
        """
        B = positions.shape[0]

        if atom_mask.dim() == 1:
            mask_2d = atom_mask.unsqueeze(0).expand(B, -1)
        else:
            mask_2d = atom_mask

        # Number of real atoms per sample
        n_real = mask_2d.sum(dim=-1)  # (B,)

        z, total_log_det = self.forward(positions, atom_types, atom_mask)

        # log p_z(z) — only over real atoms
        z_masked = z * mask_2d.unsqueeze(-1)  # (B, N, 3)
        log_pz = -0.5 * (z_masked ** 2).sum(dim=(-2, -1))  # (B,)
        log_pz = log_pz - 0.5 * n_real * 3.0 * math.log(2 * math.pi)

        # NLL = -(log p_z + log_det)
        nll = -(log_pz + total_log_det)  # (B,)

        # Per degree-of-freedom normalization for stability
        dof = n_real * 3.0  # (B,)
        nll_per_dof = nll / (dof + 1e-8)  # (B,)

        # Log-det per DOF diagnostic
        log_det_per_dof = total_log_det / (dof + 1e-8)  # (B,)

        # Log-det regularization penalty (Andrade et al. 2024)
        log_det_penalty = (log_det_per_dof ** 2).mean()

        # Total loss
        nll_loss_only = nll_per_dof.mean()
        loss = nll_loss_only + log_det_reg_weight * log_det_penalty

        info = {
            "nll": nll.mean().item(),
            "nll_per_dof": nll_per_dof.mean().item(),
            "nll_loss_only": nll_loss_only.item(),
            "log_pz": log_pz.mean().item(),
            "total_log_det": total_log_det.mean().item(),
            "log_det_per_dof": log_det_per_dof.mean().item(),
            "log_det_penalty": log_det_penalty.item(),
            "n_real_mean": n_real.float().mean().item(),
        }

        return loss, info

    @torch.no_grad()
    def sample(
        self,
        atom_types: torch.Tensor,
        atom_mask: torch.Tensor,
        n_samples: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Sample from the model by inverting the flow.

        z ~ N(0, T^2 * I) for real atoms, then apply inverse transforms.

        Args:
            atom_types: (max_atoms,) atom type indices — same for all samples
            atom_mask: (max_atoms,) 1=real, 0=padding
            n_samples: number of samples to generate
            temperature: scale of the noise (1.0 = standard Gaussian)

        Returns:
            samples: (n_samples, max_atoms, 3) generated positions
        """
        device = next(self.parameters()).device
        N = atom_mask.shape[0]

        # Sample from base distribution — only real atoms get noise
        z = torch.randn(n_samples, N, 3, device=device) * temperature
        z = z * atom_mask.to(device).unsqueeze(0).unsqueeze(-1)  # zero out padding

        # Expand atom_types and atom_mask to batch
        atom_types_b = atom_types.to(device).unsqueeze(0).expand(n_samples, -1)
        atom_mask_b = atom_mask.to(device).unsqueeze(0).expand(n_samples, -1)

        # Get atom type embeddings
        atom_type_emb = self.atom_type_emb(atom_types_b)  # (B, N, emb_dim)

        # Apply inverse blocks (and ActNorm) in reverse order
        x = z
        if self.use_actnorm:
            for block, actnorm in zip(reversed(self.blocks), reversed(self.actnorm_layers)):
                x = actnorm.inverse(x, atom_mask_b)
                x = block.inverse(x, atom_type_emb, atom_mask_b)
        else:
            for block in reversed(self.blocks):
                x = block.inverse(x, atom_type_emb, atom_mask_b)

        return x

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# EMA Model (for HEURISTICS angle — SBG recipe, Tan et al. 2025)
# =============================================================================

class EMAModel:
    """Exponential Moving Average of model parameters.

    Reference: Tan, H., Tong, A., et al. "Scalable Equilibrium Sampling with
    Sequential Boltzmann Generators," ICML 2025.

    Used during evaluation: apply_shadow() temporarily swaps model params
    to EMA weights, then restore() brings back the originals.

    Args:
        model: the nn.Module to track
        decay: EMA decay factor (default 0.999, from SBG recipe)
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
        }

    def update(self):
        """Update shadow weights: shadow = decay * shadow + (1 - decay) * param."""
        for name, param in self.model.named_parameters():
            self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply_shadow(self):
        """Temporarily replace model params with EMA shadow weights."""
        self.backup = {
            name: param.clone()
            for name, param in self.model.named_parameters()
        }
        for name, param in self.model.named_parameters():
            param.data.copy_(self.shadow[name])

    def restore(self):
        """Restore original model params (after applying shadow for eval)."""
        for name, param in self.model.named_parameters():
            param.data.copy_(self.backup[name])
