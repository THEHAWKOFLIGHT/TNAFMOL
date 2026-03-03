#
# tarflow_apple.py — Clean re-implementation of Apple TarFlow
# Reference: /home/kai_nelson/the_rig/apple_ml_tarflow/transformer_flow.py
# Part of und_001 TarFlow Diagnostic Ladder.
#
# Key differences from our model.py (documented in source_comparison.md):
#   - Per-DIMENSION scale (D scales per token, not 1 shared scalar)
#   - Output SHIFT for correct autoregression (not SOS token)
#   - NO clamping on scale
#   - NO dropout
#   - NO log-det regularization
#   - Multiple attention layers per flow block (num_layers parameter)
#   - Pre-norm LayerNorm
#   - Learned positional embeddings
#
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Permutations
# ---------------------------------------------------------------------------

class Permutation(nn.Module):
    def __init__(self, seq_length: int):
        super().__init__()
        self.seq_length = seq_length

    def forward(self, x: torch.Tensor, dim: int = 1, inverse: bool = False) -> torch.Tensor:
        raise NotImplementedError('Overload me')


class PermutationIdentity(Permutation):
    def forward(self, x: torch.Tensor, dim: int = 1, inverse: bool = False) -> torch.Tensor:
        return x


class PermutationFlip(Permutation):
    def forward(self, x: torch.Tensor, dim: int = 1, inverse: bool = False) -> torch.Tensor:
        return x.flip(dims=[dim])


# ---------------------------------------------------------------------------
# Attention (pre-norm, multi-head, with KV caching for sampling)
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    """Pre-norm multi-head attention with optional KV caching for autoregressive sampling."""

    def __init__(self, in_channels: int, head_channels: int):
        assert in_channels % head_channels == 0, \
            f"in_channels={in_channels} must be divisible by head_channels={head_channels}"
        super().__init__()
        self.norm = nn.LayerNorm(in_channels)
        self.qkv = nn.Linear(in_channels, in_channels * 3)
        self.proj = nn.Linear(in_channels, in_channels)
        self.num_heads = in_channels // head_channels
        # Scale factor: head_channels^(-0.25), applied as sqrt_scale^2/temp
        self.sqrt_scale = head_channels ** (-0.25)
        # Sampling state
        self.sample = False
        self.k_cache: dict[str, list[torch.Tensor]] = {'cond': [], 'uncond': []}
        self.v_cache: dict[str, list[torch.Tensor]] = {'cond': [], 'uncond': []}

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        temp: float = 1.0,
        which_cache: str = 'cond',
    ) -> torch.Tensor:
        B, T, C = x.size()
        # Pre-norm with float cast for stability
        x = self.norm(x.float()).type(x.dtype)
        # QKV projection → split into (B, num_heads, T, head_dim)
        q, k, v = (
            self.qkv(x)
            .reshape(B, T, 3 * self.num_heads, -1)
            .transpose(1, 2)
            .chunk(3, dim=1)
        )  # each: (B, num_heads, T, head_dim)

        if self.sample:
            # KV caching: accumulate K and V across autoregressive steps
            self.k_cache[which_cache].append(k)
            self.v_cache[which_cache].append(v)
            k = torch.cat(self.k_cache[which_cache], dim=2)  # sequence dim is 2
            v = torch.cat(self.v_cache[which_cache], dim=2)

        scale = self.sqrt_scale ** 2 / temp
        if mask is not None:
            mask = mask.bool()
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)
        x = x.transpose(1, 2).reshape(B, T, C)
        x = self.proj(x)
        return x


# ---------------------------------------------------------------------------
# MLP (pre-norm, no dropout)
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, channels: int, expansion: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.main = nn.Sequential(
            nn.Linear(channels, channels * expansion),
            nn.GELU(),
            nn.Linear(channels * expansion, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm with float cast
        return self.main(self.norm(x.float()).type(x.dtype))


# ---------------------------------------------------------------------------
# AttentionBlock = residual attention + residual MLP
# ---------------------------------------------------------------------------

class AttentionBlock(nn.Module):
    def __init__(self, channels: int, head_channels: int, expansion: int = 4):
        super().__init__()
        self.attention = Attention(channels, head_channels)
        self.mlp = MLP(channels, expansion)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        attn_temp: float = 1.0,
        which_cache: str = 'cond',
    ) -> torch.Tensor:
        x = x + self.attention(x, attn_mask, attn_temp, which_cache)
        x = x + self.mlp(x)
        return x


# ---------------------------------------------------------------------------
# MetaBlock — one normalizing flow block
# ---------------------------------------------------------------------------

class MetaBlock(nn.Module):
    """
    One flow block. Implements an autoregressive affine coupling layer using
    a transformer as the conditioner.

    Autoregressive correctness:
        - Causal attention mask: token i attends to 0..i (inclusive)
        - Output SHIFT: cat([zeros, x[:,:-1]]) so that token i's affine params
          come from transformer output at position i-1, which only saw 0..i-1
        - Token 0 always gets zero params → identity transform
    """
    attn_mask: torch.Tensor  # registered buffer

    def __init__(
        self,
        in_channels: int,
        channels: int,
        num_patches: int,
        permutation: Permutation,
        num_layers: int = 1,
        head_dim: int = 64,
        expansion: int = 4,
        nvp: bool = True,
        num_classes: int = 0,
    ):
        super().__init__()
        self.proj_in = nn.Linear(in_channels, channels)
        # Learned positional embeddings (small init)
        self.pos_embed = nn.Parameter(torch.randn(num_patches, channels) * 1e-2)
        # Optional class conditioning
        if num_classes:
            self.class_embed = nn.Parameter(torch.randn(num_classes, 1, channels) * 1e-2)
        else:
            self.class_embed = None
        # Multiple attention layers per flow block
        self.attn_blocks = nn.ModuleList(
            [AttentionBlock(channels, head_dim, expansion) for _ in range(num_layers)]
        )
        self.nvp = nvp
        output_dim = in_channels * 2 if nvp else in_channels
        self.proj_out = nn.Linear(channels, output_dim)
        # CRITICAL: zero-initialize output projection weights
        self.proj_out.weight.data.fill_(0.0)
        self.permutation = permutation
        # Causal mask: lower triangular (token i sees 0..i inclusive)
        self.register_buffer(
            'attn_mask', torch.tril(torch.ones(num_patches, num_patches))
        )

    def forward(
        self, x: torch.Tensor, y: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass (data → latent).

        Args:
            x: (B, T, D) input patches/tokens
            y: (B,) class labels (optional)

        Returns:
            z: (B, T, D) transformed latent
            logdet: (B,) per-sample log-determinant contribution (scalar per batch)
        """
        # Apply permutation to input and positional embeddings
        x = self.permutation(x)
        pos_embed = self.permutation(self.pos_embed, dim=0)

        x_in = x  # keep original for the affine transform
        x = self.proj_in(x) + pos_embed

        # Class conditioning
        if self.class_embed is not None:
            if y is not None:
                if (y < 0).any():
                    # Classifier-free guidance: mix cond and uncond
                    m = (y < 0).float().view(-1, 1, 1)
                    class_embed = (1 - m) * self.class_embed[y] + m * self.class_embed.mean(dim=0)
                else:
                    class_embed = self.class_embed[y]
                x = x + class_embed
            else:
                x = x + self.class_embed.mean(dim=0)

        # Apply attention blocks with causal mask
        for block in self.attn_blocks:
            x = block(x, self.attn_mask)

        x = self.proj_out(x)

        # CRITICAL: Output shift for correct autoregression
        # After this, params for token i come from output[i-1] (which only saw 0..i-1)
        # Token 0 gets zeros → identity transform
        x = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)

        if self.nvp:
            # NVP: split into log-scale (xa) and shift (xb) — each D-dimensional
            xa, xb = x.chunk(2, dim=-1)
        else:
            # VP: no scale, only shift
            xb = x
            xa = torch.zeros_like(x)

        # Affine transform (forward): z = (x - shift) * exp(-log_scale)
        # Note: SUBTRACT shift, MULTIPLY by exp(-xa)  — matches Apple exactly
        scale = (-xa.float()).exp().type(xa.dtype)
        z = (x_in - xb) * scale

        # Apply inverse permutation to output
        z = self.permutation(z, inverse=True)

        # Log-determinant: mean of -xa over patches and channels
        # = -xa.mean(dim=[1,2]) per sample
        logdet = -xa.mean(dim=[1, 2])

        return z, logdet

    def reverse_step(
        self,
        x: torch.Tensor,
        pos_embed: torch.Tensor,
        i: int,
        y: torch.Tensor | None = None,
        attn_temp: float = 1.0,
        which_cache: str = 'cond',
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        One autoregressive step during sampling (latent → data).
        Uses KV caching — assumes set_sample_mode(True) was called.

        Args:
            x: (B, T, D) current partially-decoded sequence (only x[:, i] will be updated)
            pos_embed: (T, channels) positional embeddings (after permutation)
            i: current token index (0-indexed)
            y: class labels

        Returns:
            xa: (B, 1, D) log-scale for position i
            xb: (B, 1, D) shift for position i
        """
        x_in = x[:, i:i + 1]  # (B, 1, D) — current token, keep seq dim
        x = self.proj_in(x_in) + pos_embed[i:i + 1]

        if self.class_embed is not None:
            if y is not None:
                x = x + self.class_embed[y]
            else:
                x = x + self.class_embed.mean(dim=0)

        # No attn_mask here — KV cache handles causality
        for block in self.attn_blocks:
            x = block(x, attn_temp=attn_temp, which_cache=which_cache)

        x = self.proj_out(x)

        if self.nvp:
            xa, xb = x.chunk(2, dim=-1)
        else:
            xb = x
            xa = torch.zeros_like(x)

        return xa, xb

    def set_sample_mode(self, flag: bool = True):
        """Enable/disable KV caching for all Attention modules in this block."""
        for m in self.modules():
            if isinstance(m, Attention):
                m.sample = flag
                m.k_cache = {'cond': [], 'uncond': []}
                m.v_cache = {'cond': [], 'uncond': []}

    def reverse(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
        guidance: float = 0,
        guide_what: str = 'ab',
        attn_temp: float = 1.0,
        annealed_guidance: bool = False,
    ) -> torch.Tensor:
        """
        Full reverse pass (latent → data) using KV-cached autoregressive sampling.

        The reverse of the affine transform z = (x - shift) * exp(-log_scale) is:
            x[i+1] = z[i+1] * exp(xa[i]) + xb[i]
        where xa[i], xb[i] are the affine params produced for position i (conditioned on 0..i-1).
        """
        x = self.permutation(x)
        pos_embed = self.permutation(self.pos_embed, dim=0)

        self.set_sample_mode(True)
        T = x.size(1)

        for i in range(T - 1):
            # Get affine params for position i+1, conditioned on 0..i
            za, zb = self.reverse_step(x, pos_embed, i, y, which_cache='cond')

            if guidance > 0 and guide_what:
                # Unconditional step for CFG
                za_u, zb_u = self.reverse_step(
                    x, pos_embed, i, None, attn_temp=attn_temp, which_cache='uncond'
                )
                g = (i + 1) / (T - 1) * guidance if annealed_guidance else guidance
                if 'a' in guide_what:
                    za = za + g * (za - za_u)
                if 'b' in guide_what:
                    zb = zb + g * (zb - zb_u)

            # Inverse affine: x[i+1] = z[i+1] * exp(xa) + xb
            scale = za[:, 0].float().exp().type(za.dtype)  # (B, D), drop seq dim
            x[:, i + 1] = x[:, i + 1] * scale + zb[:, 0]

        self.set_sample_mode(False)
        return self.permutation(x, inverse=True)


# ---------------------------------------------------------------------------
# Model — top-level TarFlow for image-like data
# ---------------------------------------------------------------------------

class TarFlowApple(nn.Module):
    """
    Apple TarFlow for image-like data.

    Handles patchification (image → token sequence) and unpatchification.
    Uses a learned prior variance buffer (relevant only in non-NVP mode).
    """
    VAR_LR: float = 0.1  # EMA rate for prior variance update
    var: torch.Tensor    # registered buffer

    def __init__(
        self,
        in_channels: int,
        img_size: int,
        patch_size: int,
        channels: int,
        num_blocks: int,
        layers_per_block: int,
        head_dim: int = 64,
        expansion: int = 4,
        nvp: bool = True,
        num_classes: int = 0,
    ):
        """
        Args:
            in_channels: number of image channels (e.g. 1 for MNIST, 3 for CIFAR-10)
            img_size: spatial size of image (H=W, e.g. 28, 32)
            patch_size: size of each patch (e.g. 4 → 4×4 patches)
            channels: hidden dimension of transformer
            num_blocks: number of flow blocks (MetaBlocks)
            layers_per_block: attention layers per MetaBlock
            head_dim: channels per attention head
            expansion: MLP expansion factor
            nvp: if True, use non-volume-preserving (NVP) flow; if False, VP
            num_classes: number of conditioning classes (0 = unconditional)
        """
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size ** 2  # dimension per patch

        permutations = [
            PermutationIdentity(self.num_patches),
            PermutationFlip(self.num_patches),
        ]

        blocks = []
        for i in range(num_blocks):
            blocks.append(
                MetaBlock(
                    in_channels=self.patch_dim,
                    channels=channels,
                    num_patches=self.num_patches,
                    permutation=permutations[i % 2],
                    num_layers=layers_per_block,
                    head_dim=head_dim,
                    expansion=expansion,
                    nvp=nvp,
                    num_classes=num_classes,
                )
            )
        self.blocks = nn.ModuleList(blocks)

        # Prior variance: ones for NVP (standard Gaussian), learnable for VP
        self.register_buffer('var', torch.ones(self.num_patches, self.patch_dim))

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert image (N, C, H, W) → sequence of patches (N, T, C*patch_size²)."""
        u = F.unfold(x, self.patch_size, stride=self.patch_size)
        return u.transpose(1, 2)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert sequence of patches (N, T, C*patch_size²) → image (N, C, H, W)."""
        u = x.transpose(1, 2)
        return F.fold(u, (self.img_size, self.img_size), self.patch_size, stride=self.patch_size)

    def forward(
        self, x: torch.Tensor, y: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        """
        Forward pass (data → latent).

        Args:
            x: (N, C, H, W) image tensor
            y: (N,) class labels (optional)

        Returns:
            z: (N, T, D) latent representation
            outputs: list of intermediate z values (one per block)
            logdets: (N,) accumulated log-determinant per sample
        """
        x = self.patchify(x)
        outputs = []
        logdets = torch.zeros(x.size(0), device=x.device)
        for block in self.blocks:
            x, logdet = block(x, y)
            logdets = logdets + logdet
            outputs.append(x)
        return x, outputs, logdets

    def update_prior(self, z: torch.Tensor):
        """EMA update of prior variance (used in VP mode)."""
        z2 = (z ** 2).mean(dim=0)
        self.var.lerp_(z2.detach(), weight=self.VAR_LR)

    def get_loss(self, z: torch.Tensor, logdets: torch.Tensor) -> torch.Tensor:
        """
        NLL loss per dimension.

        loss = 0.5 * mean(z²) - mean(logdets)

        Sanity checks:
            - For z ~ N(0,I) with logdets=0: loss = 0.5 (Gaussian entropy in nats/dim)
            - Minimizing this is equivalent to maximizing log p(x) under a Gaussian prior
            - z.pow(2).mean() averages over (B, T, D) — correct NLL per dimension
            - logdets.mean() averages over batch — correct
        """
        return 0.5 * z.pow(2).mean() - logdets.mean()

    @torch.no_grad()
    def reverse(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
        guidance: float = 0,
        guide_what: str = 'ab',
        attn_temp: float = 1.0,
        annealed_guidance: bool = False,
        return_sequence: bool = False,
    ) -> torch.Tensor | list[torch.Tensor]:
        """
        Generate samples by reversing the flow.

        Args:
            x: (N, T, D) latent samples (typically from N(0, var))
            y: class labels (optional)
            guidance: CFG guidance scale (0 = disabled)
            guide_what: which params to guide ('a', 'b', or 'ab')
            attn_temp: attention temperature for sampling
            annealed_guidance: linearly anneal guidance over tokens
            return_sequence: if True, return all intermediate images

        Returns:
            image or list of images
        """
        seq = [self.unpatchify(x)]
        # Scale latent by prior std before decoding
        x = x * self.var.sqrt()
        for block in reversed(self.blocks):
            x = block.reverse(x, y, guidance, guide_what, attn_temp, annealed_guidance)
            seq.append(self.unpatchify(x))
        x = self.unpatchify(x)

        if not return_sequence:
            return x
        else:
            return seq

    def sample(
        self,
        n: int,
        device: torch.device,
        y: torch.Tensor | None = None,
        temp: float = 1.0,
        guidance: float = 0,
        guide_what: str = 'ab',
        attn_temp: float = 1.0,
        annealed_guidance: bool = False,
    ) -> torch.Tensor:
        """
        Sample n images from the model.

        Args:
            n: number of samples
            device: target device
            y: class labels, shape (n,) — set to None for unconditional
            temp: temperature (scales the initial noise)
            guidance: CFG guidance scale
            guide_what: 'a', 'b', or 'ab'
            attn_temp: attention temperature
            annealed_guidance: linearly anneal guidance

        Returns:
            (n, C, H, W) tensor of generated images
        """
        z = torch.randn(n, self.num_patches, self.patch_dim, device=device) * temp
        return self.reverse(z, y, guidance, guide_what, attn_temp, annealed_guidance)


# ---------------------------------------------------------------------------
# TarFlow1D — variant for non-image sequential data (e.g. 2D points, molecules)
# ---------------------------------------------------------------------------

class TarFlow1D(nn.Module):
    """
    TarFlow for 1D sequence data (no patchification).

    Use this when the input is already a sequence of D-dimensional tokens.
    For 2D data: seq_length=1, in_channels=2 (single 2D point as one token).
    For molecular data: seq_length=N_atoms, in_channels=3 (Cartesian coords).

    Prior is always N(0, I) in NVP mode (standard).
    """
    var: torch.Tensor

    def __init__(
        self,
        in_channels: int,
        seq_length: int,
        channels: int,
        num_blocks: int,
        layers_per_block: int,
        head_dim: int = 64,
        expansion: int = 4,
        nvp: bool = True,
        num_classes: int = 0,
    ):
        """
        Args:
            in_channels: dimension of each token (e.g. 2 for 2D, 3 for 3D coords)
            seq_length: number of tokens in the sequence
            channels: transformer hidden dim
            num_blocks: number of flow blocks
            layers_per_block: attention layers per flow block
            head_dim: attention head dimension
            expansion: MLP expansion factor
            nvp: non-volume-preserving (True) or volume-preserving (False)
            num_classes: conditioning classes (0 = unconditional)
        """
        super().__init__()
        self.seq_length = seq_length
        self.in_channels = in_channels

        permutations = [
            PermutationIdentity(seq_length),
            PermutationFlip(seq_length),
        ]

        blocks = []
        for i in range(num_blocks):
            blocks.append(
                MetaBlock(
                    in_channels=in_channels,
                    channels=channels,
                    num_patches=seq_length,
                    permutation=permutations[i % 2],
                    num_layers=layers_per_block,
                    head_dim=head_dim,
                    expansion=expansion,
                    nvp=nvp,
                    num_classes=num_classes,
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.register_buffer('var', torch.ones(seq_length, in_channels))

    def forward(
        self, x: torch.Tensor, y: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass (data → latent).

        Args:
            x: (B, T, D) sequence of tokens
            y: (B,) class labels (optional)

        Returns:
            z: (B, T, D) latent
            logdets: (B,) log-determinant per sample
        """
        logdets = torch.zeros(x.size(0), device=x.device)
        for block in self.blocks:
            x, logdet = block(x, y)
            logdets = logdets + logdet
        return x, logdets

    def get_loss(self, z: torch.Tensor, logdets: torch.Tensor) -> torch.Tensor:
        """NLL per dimension: 0.5 * mean(z²) - mean(logdets)."""
        return 0.5 * z.pow(2).mean() - logdets.mean()

    @torch.no_grad()
    def reverse(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
        guidance: float = 0,
        guide_what: str = 'ab',
        attn_temp: float = 1.0,
        annealed_guidance: bool = False,
    ) -> torch.Tensor:
        """Reverse pass (latent → data)."""
        x = x * self.var.sqrt()
        for block in reversed(self.blocks):
            x = block.reverse(x, y, guidance, guide_what, attn_temp, annealed_guidance)
        return x

    @torch.no_grad()
    def sample(
        self,
        n: int,
        device: torch.device,
        y: torch.Tensor | None = None,
        temp: float = 1.0,
    ) -> torch.Tensor:
        """Sample n sequences from the model."""
        z = torch.randn(n, self.seq_length, self.in_channels, device=device) * temp
        return self.reverse(z, y)
