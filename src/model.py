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
- STRICTLY CAUSAL: atom at position i+1 (0-indexed, 0=SOS) attends only to positions
  0..i-1 (NOT including itself). SOS can self-attend. This gives a lower-triangular
  Jacobian (off-diagonal block), required for correct log-determinant computation.

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

hyp_004 changes:
- BidirectionalTypeEncoder: 1-layer bidirectional transformer encoder over atom type
  embeddings. Runs once per forward/inverse pass, enriches each atom's type embedding
  with full molecular composition context (no causal mask). Addresses the gap where
  causal masking hides future atom types during generation.
- TarFlowBlock: optional learned positional encoding (nn.Embedding) added in d_model
  space after input projection, before attention. Each block has independent positional
  embeddings.
- TarFlow: use_bidir_types flag to toggle bidirectional type encoder, use_pos_enc flag
  to toggle positional encodings.

hyp_005 changes:
- CAUSAL MASK BUG FIX: _build_causal_mask() now uses strictly causal masking.
  Previously `allowed[i, :i+1] = True` was self-inclusive (atom at position i+1
  attended to itself). Now: SOS [0,0]=True, atoms [i, :i]=True for i>=1.
  This restores the lower-triangular Jacobian required for correct NLL computation.
  (Bug was present since hyp_002; confirmed harmful in und_001.)
- zero_padding_queries: TarFlowBlock accepts zero_padding_queries=True flag.
  When enabled, zeroes the d_model-projected query of all padding atoms before attention.
  This prevents padding from corrupting LayerNorm statistics and gradient flow — one of
  the two padding corruption channels identified in und_001.

hyp_006 changes:
- use_output_shift: TarFlowBlock and TarFlow accept use_output_shift=True flag.
  When enabled, replaces the SOS+strictly-causal-mask mechanism with Apple's output-shift
  mechanism (Salimans & Ho 2021, as implemented in Apple TarFlow):
    * No SOS token — transformer runs on N tokens directly
    * Self-inclusive causal mask: token i attends to 0..i (lower triangular with diagonal)
    * Output shift: after proj_out, shift output by one position:
        params = cat([zeros_like(params[:,:1,:]), params[:,:-1,:]], dim=1)
      This means params for token i come from transformer output at position i-1.
      Token 0 gets zero params → identity transform. HARD autoregressive guarantee.
    * out_proj zero-initialized for stable start
  Hypothesis: the SOS+strictly-causal mechanism creates an exploitation pathway not
  present in output-shift architecture. Output-shift eliminates this pathway, enabling
  multi-molecule training without log-det explosion.
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
# Bidirectional Type Encoder (hyp_004)
# =============================================================================

class BidirectionalTypeEncoder(nn.Module):
    """Encode atom types bidirectionally — every atom sees all other types.

    Single transformer encoder layer with NO causal mask. Produces per-atom
    context vectors that encode the full molecular composition.

    This addresses a key architectural gap: in the standard TarFlow, causal
    masking means atom i only sees types of atoms 0..i-1. During generation,
    later atoms are built without knowing what atom types come next. The
    bidirectional encoder computes a composition-aware embedding ONCE, then
    passes it to all blocks as conditioning context.

    Args:
        emb_dim: dimension of atom type embeddings (must match TarFlow.atom_type_emb_dim)
        n_heads: number of attention heads (default 2)
        ffn_dim: feedforward hidden dimension (default 64)
        dropout: dropout rate (default 0.1)
    """

    def __init__(self, emb_dim: int = 16, n_heads: int = 2, ffn_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            batch_first=True,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, atom_type_emb: torch.Tensor, atom_mask: torch.Tensor) -> torch.Tensor:
        """Bidirectional encoding of atom type embeddings.

        Args:
            atom_type_emb: (B, N, emb_dim) raw atom type embeddings
            atom_mask: (B, N) 1=real, 0=padding

        Returns:
            type_context: (B, N, emb_dim) enriched per-atom context vectors
        """
        # key_padding_mask: True where position should be IGNORED
        key_padding_mask = (atom_mask < 0.5)  # (B, N) True=padding
        out = self.encoder(atom_type_emb, src_key_padding_mask=key_padding_mask)
        out = self.norm(out)
        # Zero out padding positions
        out = out * atom_mask.unsqueeze(-1)
        return out  # (B, N, emb_dim)


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
        use_pos_enc: if True, add learned positional encodings (hyp_004)
        max_atoms: max number of atoms for positional encoding table (default 21)
        zero_padding_queries: if True, zero the d_model query of padding atoms before
            attention (hyp_005). Prevents padding from corrupting LayerNorm statistics
            and gradient flow. Applied after input_proj + pos_enc, before attention.
        use_output_shift: if True, use Apple's output-shift autoregressive mechanism
            instead of SOS+strictly-causal-mask (hyp_006). No SOS token; transformer
            runs on N tokens with self-inclusive causal mask; params shifted by 1 position
            after out_proj. Token 0 gets zero params = identity transform.
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
        use_pos_enc: bool = False,
        max_atoms: int = 21,
        zero_padding_queries: bool = False,
        use_output_shift: bool = False,
        log_scale_max: float = 0.5,  # DEPRECATED — ignored, kept for compat
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.reverse = reverse
        self.alpha_pos = alpha_pos
        self.alpha_neg = alpha_neg
        self.shift_only = shift_only
        self.use_pos_enc = use_pos_enc
        self.zero_padding_queries = zero_padding_queries
        self.use_output_shift = use_output_shift

        # Learnable SOS token (provides context for the first atom)
        # Only used when use_output_shift=False (SOS path).
        if not use_output_shift:
            self.sos = nn.Parameter(torch.randn(1, 1, d_model) * 0.01)

        # Input projection: in_features -> d_model
        self.input_proj = nn.Linear(in_features, d_model)

        # Positional encodings (hyp_004): learned embedding per sequence position
        # SOS path: max_atoms + 1 entries (SOS at position 0, atoms at 1..N)
        # Output-shift path: max_atoms entries only (atoms at 0..N-1, no SOS)
        if use_pos_enc:
            pos_enc_size = max_atoms if use_output_shift else max_atoms + 1
            self.pos_embed = nn.Embedding(pos_enc_size, d_model)

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

        # Initialize output projection near zero for stable start.
        # For use_output_shift=True: zero-initialize weights (Apple's convention — bias stays zero).
        # For use_output_shift=False: zero-initialize both weights and bias (existing behavior).
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def _build_causal_mask(self, n_atoms: int, device) -> torch.Tensor:
        """Build strictly causal attention mask for n_atoms + 1 (SOS) positions.

        For forward order: atom i (at position i+1 in the sequence, SOS is at 0)
        attends to [SOS, atom_0, ..., atom_{i-1}] = positions [0, 1, ..., i-1].
        This is STRICTLY CAUSAL: atom at position i+1 does NOT attend to itself.

        Special case: SOS at position 0 can self-attend (it has no prior context).

        So in the full (N+1) x (N+1) attention matrix:
          row 0 (SOS): can attend to column 0 only (self)
          row i+1 (atom i): can attend to columns [0, 1, ..., i-1] (SOS + prior atoms, NOT self)

        This gives the correct lower-triangular Jacobian structure: y_i = f(x_0, ..., x_{i-1})
        so dy_i/dx_i is determined only by the direct affine transform (exp(log_scale_i)),
        making the Jacobian lower-triangular with log-det = sum_i 3*log_scale_i.

        For reverse order: atoms are passed in reversed order externally, so the same
        strictly causal mask applies to the reversed sequence.

        Sanity checks:
        - Row 0 (SOS): allows column 0 only — SOS sees only itself. CHECK.
        - Row 1 (atom 0): allows column 0 only — atom 0 sees only SOS. CHECK.
        - Row i+1 (atom i): allows columns 0..i-1 — atom i sees SOS + atoms 0..i-1. CHECK.
        - No row i+1 allows column i+1 (self) — strictly causal. CHECK.
        - Jacobian: y_i = exp(log_scale_i(x_{<i})) * x_i + shift_i(x_{<i})
          => dy_i/dx_i = exp(log_scale_i) — diagonal element of Jacobian block. CHECK.
          => dy_i/dx_j = 0 for j > i — upper-triangular block is zero. CHECK.

        Args:
            n_atoms: number of atoms (not including SOS)
            device: torch device

        Returns:
            attn_bias: (N+1, N+1) additive bias — 0 for allowed, -inf for masked
        """
        N1 = n_atoms + 1  # +1 for SOS
        # Create allowed mask: (N1, N1) bool, True = allowed
        allowed = torch.zeros(N1, N1, dtype=torch.bool, device=device)

        # SOS at position 0 self-attends
        allowed[0, 0] = True

        # Atom at position i+1 (for i in 0..N-1) attends to positions 0..i (strictly causal: NOT i+1)
        # positions 0..i = SOS + atoms 0..i-1
        for i in range(1, N1):
            allowed[i, :i] = True

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

        # Add positional encodings (hyp_004) — applied BEFORE attention, in d_model space
        # Positions are in canonical order (0=SOS, 1..N=atoms).
        # For reverse blocks, the atom sequence was already flipped externally,
        # so positional encoding encodes atom *identity* (atom 0 is always atom 0
        # in the original ordering, regardless of block direction).
        if self.use_pos_enc:
            pos_indices = torch.arange(N + 1, device=device)  # 0..N (SOS + atoms)
            h = h + self.pos_embed(pos_indices)  # broadcast: (B, N+1, d_model)

        # Zero padding queries (hyp_005): zero the d_model activation of padding atoms
        # Applied after input_proj + pos_enc, before attention.
        # This prevents padding atoms from influencing attention queries and corrupting
        # LayerNorm statistics. SOS (position 0) is always real — only atom positions
        # 1..N+1 are potentially padding (h_mask uses atom_mask for positions 1..N).
        # Sanity check: SOS at position 0 is never zeroed (sos_ones = 1). Padding
        # atom positions (atom_mask=0) get h[:, that_position, :] = 0.
        if self.zero_padding_queries:
            sos_ones = torch.ones(B, 1, device=device)
            h_mask = torch.cat([sos_ones, atom_mask.float()], dim=1)  # (B, N+1)
            h = h * h_mask.unsqueeze(-1)  # (B, N+1, d_model) — zero padding positions

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

    def _run_transformer_output_shift(
        self,
        positions: torch.Tensor,
        atom_type_emb: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run transformer WITHOUT SOS, using self-inclusive causal mask (hyp_006).

        Autoregressive correctness is enforced by the output shift AFTER this method
        returns, not by the causal mask. The self-inclusive mask (token i attends to 0..i)
        is correct here because the output shift will discard the i-th output for atom i's
        parameters — instead using output i-1 (which only saw tokens 0..i-1).

        Causal mask structure (N x N, self-inclusive lower triangular):
            Token 0: attends to [0]           → params come from output[-1]=zeros → identity
            Token 1: attends to [0, 1]         → params come from output[0] → conditioned on x_0
            Token i: attends to [0..i]         → params come from output[i-1] → conditioned on x_{<i}

        Key differences from _run_transformer():
          - No SOS token: input shape is (B, N, d_model) not (B, N+1, d_model)
          - Self-inclusive causal mask: torch.tril(ones(N, N)) instead of strictly causal
          - Padding key mask: same logic (mask padding atoms as keys), but over N positions
          - zero_padding_queries: applied to all N atom positions (no SOS to protect)
          - pos_embed: N entries (not N+1)

        Sanity checks:
          - Token 0 attends to itself only — its output feeds params for token 1. CHECK.
          - After output shift, params[:,0,:] = zeros (cat of zeros_like). CHECK.
          - After output shift, params[:,i,:] = transformer_output[:,i-1,:]. CHECK.
          - Params for token i come from output at i-1 → conditioned on x_{0..i-1}. CHECK.
          - dy_i/dx_j = 0 for j > i — lower-triangular Jacobian preserved. CHECK.

        Args:
            positions: (B, N, 3) in causal ordering (already flipped for reverse blocks)
            atom_type_emb: (B, N, emb_dim) in causal ordering
            atom_mask: (B, N) in causal ordering

        Returns:
            atom_out: (B, N, d_model) — transformer output per atom (before output shift)
        """
        B, N, _ = positions.shape
        device = positions.device

        # Project input directly (no SOS prepended)
        features = torch.cat([positions, atom_type_emb], dim=-1)  # (B, N, 3+emb_dim)
        h = self.input_proj(features)  # (B, N, d_model)

        # Positional encodings (only if enabled): N entries, no SOS
        if self.use_pos_enc:
            pos_indices = torch.arange(N, device=device)  # 0..N-1
            h = h + self.pos_embed(pos_indices)  # broadcast: (B, N, d_model)

        # Zero padding queries (hyp_005 + hyp_006 compatible): zero all padding atom activations
        # No SOS to protect here — apply directly to all N positions
        if self.zero_padding_queries:
            h = h * atom_mask.float().unsqueeze(-1)  # (B, N, d_model)

        # Self-INCLUSIVE causal mask: token i attends to 0..i (lower triangular with diagonal)
        # This is correct because output shift will use output[i-1] for params[i],
        # so position i's self-attention output (which sees x_0..x_i) is used for params[i+1].
        causal_mask_bool = torch.tril(torch.ones(N, N, dtype=torch.bool, device=device))
        causal_bias = torch.zeros(N, N, device=device)
        causal_bias[~causal_mask_bool] = float("-inf")  # mask upper triangle

        # Padding key mask: True where position should be ignored as key (B, N) bool
        atom_kpm = (atom_mask < 0.5)  # (B, N) True=padding

        # Combine causal bias and padding key mask into single float additive mask
        # combined: (B, N, N) — additive bias per batch element
        combined_mask = causal_bias.unsqueeze(0).expand(B, -1, -1).clone()  # (B, N, N)
        padding_cols = atom_kpm.unsqueeze(1).float() * -1e9  # (B, 1, N)
        combined_mask = combined_mask + padding_cols  # broadcast: (B, N, N)

        # Expand for all heads: (B*n_heads, N, N)
        combined_mask = combined_mask.unsqueeze(1).expand(
            -1, self.n_heads, -1, -1
        ).reshape(B * self.n_heads, N, N)

        # Self-attention
        h_attn, _ = self.attn(h, h, h, attn_mask=combined_mask)
        h = self.attn_norm(h + self.attn_dropout(h_attn))

        # FFN
        h = self.ffn_norm(h + self.ffn(h))

        return h  # (B, N, d_model) — output shift applied in forward()

    def forward(
        self,
        positions: torch.Tensor,
        atom_type_emb: torch.Tensor,
        atom_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: transform positions and compute log-det.

        Two code paths controlled by self.use_output_shift:
          - SOS path (use_output_shift=False): existing mechanism — SOS token prepended,
            strictly causal mask, transformer output used directly as params.
          - Output-shift path (use_output_shift=True): Apple's mechanism — no SOS,
            self-inclusive causal mask, params shifted by 1 position after out_proj.

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
            pos_ordered = positions.flip(1)
            emb_ordered = atom_type_emb.flip(1)
            mask_ordered = atom_mask.flip(1)
        else:
            pos_ordered = positions
            emb_ordered = atom_type_emb
            mask_ordered = atom_mask

        if self.use_output_shift:
            # --- Output-shift path (hyp_006) ---
            # Run transformer with self-inclusive causal mask on N tokens (no SOS)
            atom_out = self._run_transformer_output_shift(
                pos_ordered, emb_ordered, mask_ordered
            )  # (B, N, d_model) in ordered space

            # Predict affine params in ordered space
            params = self.out_proj(atom_out)  # (B, N, 3 or 4)

            # OUTPUT SHIFT: shift params by one position.
            # params[:,i,:] becomes transformer_output[:,i-1,:] (conditioned on x_{0..i-1})
            # params[:,0,:] = zeros (identity transform for token 0).
            # This is the HARD autoregressive guarantee — correct regardless of attention mask.
            params = torch.cat([
                torch.zeros_like(params[:, :1, :]),  # (B, 1, out_dim) — zeros for token 0
                params[:, :-1, :],                    # (B, N-1, out_dim) — shifted params
            ], dim=1)  # (B, N, out_dim)

            shift = params[..., :3]  # (B, N, 3) in ordered space

            if self.shift_only:
                y_ordered = pos_ordered + shift
                log_det = torch.zeros(B, device=positions.device)
            else:
                log_scale = params[..., 3:4]  # (B, N, 1) in ordered space

                # Asymmetric soft clamp via arctan (Andrade et al. 2024)
                log_scale = _asymmetric_clamp(log_scale, self.alpha_pos, self.alpha_neg)

                # Apply affine transform in ordered space: y_i = exp(log_scale_i) * x_i + shift_i
                scale = log_scale.exp()  # (B, N, 1)
                y_ordered = scale * pos_ordered + shift  # (B, N, 3)

                # Log-determinant: each real atom contributes 3 * log_scale_i
                # Compute in ordered space using ordered mask
                log_scale_sq = log_scale.squeeze(-1)  # (B, N)
                log_det = (3.0 * log_scale_sq * mask_ordered).sum(dim=-1)  # (B,)

            # Zero padding in ordered space
            y_ordered = y_ordered * mask_ordered.unsqueeze(-1)

            # Flip back to original ordering if needed
            if self.reverse:
                y = y_ordered.flip(1)
            else:
                y = y_ordered

        else:
            # --- SOS path (original mechanism, unchanged) ---
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

        Two code paths controlled by self.use_output_shift:
          - SOS path: existing mechanism — run transformer at each step, use output[step] directly.
          - Output-shift path: at step i, run full transformer on x_work, apply output shift
            to get params, use params[i] (which is transformer_output[i-1]) to recover x_work[i].
            Step 0 special: params[0] = zeros (identity) → x_work[0] = y_work[0].

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
            # Working in ordered (reversed) space
            y_work = y.flip(1)
            emb_work = atom_type_emb.flip(1)
            mask_work = atom_mask.flip(1)
        else:
            y_work = y
            emb_work = atom_type_emb
            mask_work = atom_mask

        x_work = torch.zeros_like(y_work)

        if self.use_output_shift:
            # --- Output-shift inverse path ---
            # Step 0: params[0] = zeros (output shift guarantees this) → identity transform
            #   x_work[0] = y_work[0]
            x_work[:, 0, :] = y_work[:, 0, :]

            # Steps 1..N-1: run transformer on x_work (partially recovered),
            # apply output shift, use params[step] to recover x_work[step].
            # params[step] = transformer_output[step-1] — conditioned on x_work[0..step-1].
            # Since x_work[0..step-1] are already correctly recovered, this is causal.
            for step in range(1, N):
                atom_out = self._run_transformer_output_shift(
                    x_work, emb_work, mask_work
                )  # (B, N, d_model)

                # Apply output shift: params[:,i,:] = atom_out[:,i-1,:]
                params_raw = self.out_proj(atom_out)  # (B, N, out_dim)
                params = torch.cat([
                    torch.zeros_like(params_raw[:, :1, :]),
                    params_raw[:, :-1, :],
                ], dim=1)  # (B, N, out_dim)

                shift_step = params[:, step, :3]  # (B, 3)

                if self.shift_only:
                    x_work[:, step, :] = y_work[:, step, :] - shift_step
                else:
                    log_scale_step = params[:, step, 3:4]  # (B, 1)
                    log_scale_step = _asymmetric_clamp(log_scale_step, self.alpha_pos, self.alpha_neg)
                    scale_step = log_scale_step.exp()  # (B, 1)
                    x_work[:, step, :] = (y_work[:, step, :] - shift_step) / scale_step

        else:
            # --- SOS inverse path (original mechanism, unchanged) ---
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
        use_bidir_types: if True, use BidirectionalTypeEncoder (hyp_004)
        use_pos_enc: if True, add learned positional encodings per block (hyp_004)
        zero_padding_queries: if True, zero padding atom queries before attention (hyp_005)
        use_output_shift: if True, use Apple's output-shift mechanism (hyp_006)
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
        use_bidir_types: bool = False,
        use_pos_enc: bool = False,
        zero_padding_queries: bool = False,
        use_output_shift: bool = False,
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
        self.use_bidir_types = use_bidir_types
        self.use_pos_enc = use_pos_enc
        self.zero_padding_queries = zero_padding_queries
        self.use_output_shift = use_output_shift

        # Atom type embedding (shared across all blocks)
        self.atom_type_emb = nn.Embedding(n_atom_types, atom_type_emb_dim)

        # Bidirectional type encoder (hyp_004): enriches atom type embeddings
        # with full molecular composition context via bidirectional attention
        if use_bidir_types:
            self.type_encoder = BidirectionalTypeEncoder(
                emb_dim=atom_type_emb_dim,
                n_heads=2,
                ffn_dim=64,
                dropout=dropout,
            )
        else:
            self.type_encoder = None

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
                use_pos_enc=use_pos_enc,
                max_atoms=max_atoms,
                zero_padding_queries=zero_padding_queries,
                use_output_shift=use_output_shift,
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
        """Ensure all inputs are (B, N, ...) shaped and compute type context.

        If use_bidir_types is True, the atom type embeddings are enriched
        with full molecular composition context via the BidirectionalTypeEncoder.
        The enriched embeddings have the same shape as raw embeddings (B, N, emb_dim),
        so no downstream shape changes are needed.

        Returns:
            positions: (B, N, 3)
            atom_type_emb: (B, N, emb_dim) — raw or bidirectionally enriched
            atom_mask: (B, N)
        """
        B = positions.shape[0]

        if atom_mask.dim() == 1:
            atom_mask = atom_mask.unsqueeze(0).expand(B, -1)
        if atom_types.dim() == 1:
            atom_types = atom_types.unsqueeze(0).expand(B, -1)

        atom_type_emb = self.atom_type_emb(atom_types)  # (B, N, emb_dim)

        # Bidirectional type encoding (hyp_004): compute once, share across all blocks
        if self.use_bidir_types and self.type_encoder is not None:
            atom_type_emb = self.type_encoder(atom_type_emb, atom_mask)

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

        # Get atom type embeddings (with optional bidirectional encoding)
        atom_type_emb = self.atom_type_emb(atom_types_b)  # (B, N, emb_dim)
        if self.use_bidir_types and self.type_encoder is not None:
            atom_type_emb = self.type_encoder(atom_type_emb, atom_mask_b)

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
