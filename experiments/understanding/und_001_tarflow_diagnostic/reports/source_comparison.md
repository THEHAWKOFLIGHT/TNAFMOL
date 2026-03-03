# Source Comparison — Apple TarFlow vs. TNAFMOL TarFlow

**Phase 1 Report — und_001**
**Date:** 2026-03-02

## Files Compared

| Implementation | File | Purpose |
|----------------|------|---------|
| Apple TarFlow | `/home/kai_nelson/the_rig/apple_ml_tarflow/transformer_flow.py` | Core architecture + loss |
| Apple TarFlow | `/home/kai_nelson/the_rig/apple_ml_tarflow/train.py` | Training loop, noise augmentation, prior update |
| TNAFMOL | `src/model.py` | TarFlow + TarFlowBlock + ActNorm + EMAModel |

---

## Difference Inventory

### 1. Scale Parameterization — CRITICAL

**Apple:** Per-dimension scale. Each token/patch has D separate log-scale values (one per channel dimension). The output is split into two D-dimensional halves: `xa` (log-scale) and `xb` (shift). The transform applies element-wise: each dimension of the input gets its own independent scale factor.

```python
# Apple MetaBlock.forward
xa, xb = x.chunk(2, dim=-1)          # xa: (B, T, D), xb: (B, T, D)
scale = (-xa.float()).exp()            # D separate scales per token
y = (x_in - xb) * scale               # element-wise: each dim scaled independently
```

For molecular analogy: if a token represents an atom with 3 coordinates, Apple would predict 3 separate scales (one per coordinate).

**Ours:** Shared (isotropic) scale. Each atom gets a single scalar log-scale value applied identically to all 3 Cartesian coordinates.

```python
# Our TarFlowBlock.forward
log_scale = params[..., 3:4]           # (B, N, 1) — single scalar per atom
scale = log_scale.exp()                # (B, N, 1) broadcasts across 3 dims
y = scale * positions + shift          # same scale for x, y, z
```

**Impact:** Shared scale forces the flow to apply the same expansion/contraction factor to all 3 spatial dimensions of each atom. This severely limits the model's ability to selectively reshape distributions along different axes. It also concentrates all scale freedom into one parameter, making that parameter a single high-leverage target for log-det exploitation. Per-dimension scale distributes log-det contributions across D parameters, each with smaller individual impact on the total log-det.

**Quantitative:** With per-dim scale (D=3 per atom), the log-det contribution per atom is `log_s_x + log_s_y + log_s_z`. With shared scale, it is `3 * log_s`. The shared version has 1/3 the parameters but the single parameter has 3x the leverage on log-det, making it both less expressive AND more prone to exploitation.

---

### 2. Causal Autoregressive Mechanism — CRITICAL (Correctness Bug)

**Apple:** Standard GPT-style causal mask (position i attends to positions 0..i, inclusive of self) combined with an explicit output shift. The shift ensures that position i+1's affine parameters come from the transformer output at position i, which only saw positions 0..i. Position 0 gets zero output (identity transform).

```python
# Apple MetaBlock.forward
for block in self.attn_blocks:
    x = block(x, self.attn_mask)       # causal mask: pos i sees 0..i
x = self.proj_out(x)
x = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)  # SHIFT!
# After shift: params for pos i come from output[i-1] which saw 0..i-1
# Params for pos 0 = zeros (identity)
```

This correctly ensures: affine params for position i depend only on positions 0..i-1. The Jacobian is triangular.

**Ours:** SOS (start-of-sequence) token prepended, with a standard lower-triangular causal mask over SOS+atoms. No output shift. The mask allows position j+1 (atom j) to attend to positions 0..j+1, which INCLUDES atom j itself (column j+1).

```python
# Our _build_causal_mask
for i in range(N1):           # N1 = n_atoms + 1 (SOS + atoms)
    allowed[i, :i + 1] = True # position i sees 0..i (inclusive!)

# Consequence: atom j at position j+1 sees [SOS, atom_0, ..., atom_j]
# Atom j sees ITS OWN INPUT through the attention mechanism
```

The affine params for atom j are: `f(SOS, x_0, x_1, ..., x_j)` — depends on x_j!

**This breaks autoregression.** The Jacobian is NOT triangular because the affine transform for x_j depends on x_j through the attention mechanism. The log-det we compute (3 * log_scale per atom) assumes a triangular Jacobian and is therefore incorrect.

The actual Jacobian diagonal element for atom j is:

```
∂y_j/∂x_j = exp(s_j) * [I + x_j ⊗ ∂s_j/∂x_j] + ∂t_j/∂x_j
```

which is NOT simply `exp(s_j) * I`.

**Practical impact:** The model trains and produces some valid samples (18.3% VF at best) because:
1. Output projection is zero-initialized, so ∂s_j/∂x_j ≈ 0 initially
2. The self-dependence term may remain small relative to the direct path
3. The model optimizes a biased NLL — it's learning a useful transform but with incorrect density estimation

**To fix:** Either (a) use Apple's shift approach, or (b) change the mask to strictly-before:
```python
allowed[0, 0] = True             # SOS sees itself
for i in range(1, N1):
    allowed[i, :i] = True        # atom j at pos j+1 sees 0..j (NOT j+1)
```

---

### 3. Scale Clamping — CRITICAL

**Apple:** No clamping whatsoever. The raw network output `xa` is exponentiated directly.

```python
# Apple
scale = (-xa.float()).exp()   # raw exponential, no bounds
```

**Ours:** Asymmetric arctan soft clamping (Andrade et al. 2024) applied to log_scale before exponentiation.

```python
# Ours
log_scale = _asymmetric_clamp(log_scale, alpha_pos=0.1, alpha_neg=2.0)
# Bounded: log_scale ∈ (-alpha_neg, alpha_pos) = (-2.0, 0.1)
```

**Impact:** Apple doesn't need clamping because (a) per-dimension scale distributes log-det across many parameters, reducing per-parameter exploitation incentive, and (b) Gaussian noise augmentation smooths the target distribution, reducing the model's motivation to create sharp, degenerate transforms. Our model adopted clamping as a fix for the log-det exploitation observed in hyp_002, but this created the alpha_pos saturation equilibrium observed in hyp_003.

**Key insight for this diagnostic:** The question is whether Apple's per-dim scale + noise augmentation naturally prevents exploitation without needing clamping. If so, our clamping is treating a symptom (exploitation) rather than the root cause (shared scale + no noise augmentation).

---

### 4. Log-det Regularization — HIGH

**Apple:** None.

**Ours:** Optional quadratic penalty on log-det-per-DOF:
```python
log_det_penalty = (log_det_per_dof ** 2).mean()
loss = nll_per_dof.mean() + log_det_reg_weight * log_det_penalty
```

**Impact:** This was added in hyp_003 to combat log-det exploitation. It penalizes any deviation of log_det/dof from zero, which effectively prevents the model from learning non-trivial scale transforms. Combined with asymmetric clamping, it creates the alpha_pos saturation equilibrium.

---

### 5. Gaussian Noise Augmentation — HIGH

**Apple:** Gaussian noise sigma=0.05 added to input data every training step.

```python
# Apple train.py
if args.noise_type == 'gaussian':
    eps = args.noise_std * torch.randn_like(x)   # noise_std=0.05
    x = x + eps
```

**Ours:** None.

**Impact:** Noise augmentation smooths the target distribution, preventing the model from learning delta functions or overly sharp density estimates. For normalizing flows, this is particularly important because the model must assign positive density everywhere in the support — a degenerate transform can "cheat" by collapsing density into a small region. Noise prevents this by ensuring the target has a smooth, spread-out density. This likely reduces the incentive for log-det exploitation.

---

### 6. Learned Prior Variance — MEDIUM

**Apple:** EMA-tracked prior variance buffer. In NVP mode (default), the variance stays at ones (never updated). In non-NVP mode, updates via:
```python
# Apple
def update_prior(self, z):
    z2 = (z**2).mean(dim=0)
    self.var.lerp_(z2.detach(), weight=0.1)  # VAR_LR = 0.1

# During sampling:
x = x * self.var.sqrt()  # scale by learned prior std
```

**Ours:** Fixed N(0, I) prior. No variance tracking.

**Impact:** In NVP mode (both Apple's default and our usage), this is a non-difference — both use N(0, I). If we later switch to volume-preserving (VP) mode, we would need learned prior variance.

---

### 7. Log-det Computation Method — MEDIUM

**Apple:** Returns the MEAN of per-element log-scale values:
```python
# Apple MetaBlock.forward
return ..., -xa.mean(dim=[1, 2])  # mean over (patches, channels) → (B,)
```

Combined with `get_loss`:
```python
loss = 0.5 * z.pow(2).mean() - logdets.mean()
# = mean over (B, T, D) of 0.5*z² - mean over B of logdets
```

This is NLL per dimension, averaged over batch. Mathematically correct but uses mean normalization.

**Ours:** Returns the SUM of per-atom log-det contributions, then normalizes by DOF in the loss:
```python
# Our TarFlowBlock.forward
log_det = (3.0 * log_scale_sq * atom_mask).sum(dim=-1)  # sum over atoms → (B,)

# Our TarFlow.nll_loss
nll_per_dof = nll / (n_real * 3.0)
loss = nll_per_dof.mean()
```

**Impact:** Mathematically equivalent when there's no padding (both compute NLL per dimension per sample). With padding, ours correctly normalizes by actual DOF (n_real * 3) rather than total dimensions. This is the correct approach for variable-size molecules.

However, note that our log-det is 3*s per atom (shared scale → multiply by 3), while Apple's is the mean of D separate xa values. These represent the same mathematical quantity differently due to the per-dim vs. shared scale difference (#1).

---

### 8. Transformer Architecture — MEDIUM-HIGH

#### 8a. Layers Per Flow Block

**Apple:** Multiple attention layers per flow block (`num_layers` parameter, typically 8).
```python
# Apple MetaBlock
self.attn_blocks = ModuleList([AttentionBlock(...) for _ in range(num_layers)])
for block in self.attn_blocks:
    x = block(x, self.attn_mask)
```

Typical config: 4-8 flow blocks × 8 attention layers each = 32-64 total attention layers.

**Ours:** Single attention + FFN layer per flow block.
```python
# Our TarFlowBlock._run_transformer
h_attn, _ = self.attn(h, h, h, attn_mask=combined_mask)
h = self.attn_norm(h + self.attn_dropout(h_attn))
h = self.ffn_norm(h + self.ffn(h))
```

Typical config: 8 flow blocks × 1 attention layer each = 8 total attention layers.

**Impact:** Apple's design puts more expressive power per flow block. Each block can model complex dependencies before committing to an affine transform. Our design has more flow blocks (and therefore more affine transforms) but less capacity per block. For equivalent total attention layers, Apple can produce better affine parameters because each block's predictions are refined through multiple attention rounds.

#### 8b. Normalization: Pre-norm vs. Post-norm

**Apple:** Pre-norm (LayerNorm before attention/MLP, standard in modern transformers):
```python
x = x + attention(layernorm(x))
x = x + mlp(layernorm(x))
```

**Ours:** Post-norm (LayerNorm after residual add):
```python
h = layernorm(h + dropout(attention(h)))
h = layernorm(h + ffn(h))
```

**Impact:** Pre-norm is generally more stable for deep networks and is the modern default. Post-norm can cause training instability at depth. With our single-layer blocks this may not matter, but it's a systematic difference.

#### 8c. Dropout

**Apple:** No dropout anywhere.

**Ours:** dropout=0.1 in attention and FFN.

**Impact:** Dropout in normalizing flows is problematic. The flow's forward and inverse must be deterministic for the change-of-variables formula to hold. Dropout during training means the model learns a different effective transform than what's used at eval time (when dropout is off). This creates a train/eval mismatch that biases the learned density. For Apple's image domain, they avoid this entirely. For our molecular domain, we added it for regularization but it may be actively harmful.

---

### 9. Positional Embeddings — MEDIUM

**Apple:** Learned positional embeddings added after input projection:
```python
# Apple MetaBlock
self.pos_embed = Parameter(randn(num_patches, channels) * 1e-2)
x = self.proj_in(x) + pos_embed
```

Positional embeddings are permuted together with the sequence when direction alternates.

**Ours:** No positional embeddings. Position information comes only from the causal mask structure and the input atom coordinates.

**Impact:** Without positional embeddings, the transformer has no explicit knowledge of which atom position it's processing (beyond position-specific content in the input). In the molecular setting, atom ordering is arbitrary (unlike image patches which have spatial structure), so the benefit of positional embeddings is less clear. However, they still provide the model with an index into the autoregressive sequence, which helps it learn position-dependent transforms. Their absence may limit expressivity.

---

### 10. Conditioning Mechanism — LOW-MEDIUM

**Apple:** Class embedding as additive bias on hidden representation:
```python
# Apple MetaBlock
self.class_embed = Parameter(randn(num_classes, 1, channels) * 1e-2)
x = x + self.class_embed[y]
```

Added after input projection + positional embedding, before attention layers.

**Ours:** Atom type embedding concatenated to input features before projection:
```python
# Our TarFlowBlock._get_context_features
features = torch.cat([positions, atom_type_emb], dim=-1)  # (B, N, 3+emb_dim)
h_atoms = self.input_proj(features)                         # (B, N, d_model)
```

**Impact:** Both approaches are standard. Apple's additive conditioning is applied in hidden space (after projection), while ours is concatenated in input space (before projection). Functionally similar — the input projection can learn to combine them. Apple also supports classifier-free guidance via label dropout (not relevant for our molecular application).

---

### 11. Mixed Precision — LOW

**Apple:** Uses `.float()` cast in LayerNorm for numerical stability, casts back to input dtype:
```python
x = self.norm(x.float()).type(x.dtype)
```

Training uses GradScaler for AMP (automatic mixed precision) with bfloat16.

**Ours:** Full float32 throughout. No mixed precision.

**Impact:** Minor efficiency difference. float32 is safer for numerical stability. Apple's approach is optimized for speed on large image models.

---

### 12. Sampling / Inverse Efficiency — LOW (Inference Only)

**Apple:** KV-caching during autoregressive sampling. Each token is generated using cached key/value tensors from previous tokens, avoiding redundant computation:
```python
# Apple Attention
if self.sample:
    self.k_cache[which_cache].append(k)
    self.v_cache[which_cache].append(v)
    k = torch.cat(self.k_cache[which_cache], dim=2)
    v = torch.cat(self.v_cache[which_cache], dim=2)
```

**Ours:** Full transformer forward pass for every atom during sequential inverse:
```python
# Our TarFlowBlock.inverse
for step in range(N):
    atom_out = self._run_transformer(x_work, emb_work, mask_work)  # full N+1 sequence
```

**Impact:** Our inverse is O(N) times slower per block. For N=21 atoms this is noticeable but not a training bottleneck (inverse is only needed for sampling, not training).

---

### 13. Classifier-Free Guidance — LOW (Not Applicable)

**Apple:** Supports CFG for conditional generation:
```python
za_u, zb_u = self.reverse_step(x, pos_embed, i, None, ...)
za = za + guidance * (za - za_u)
```

**Ours:** Not implemented (not relevant for molecular generation).

**Impact:** None for our application.

---

## Summary Table

| # | Feature | Apple TarFlow | Our Implementation | Severity | Phase 3 Test |
|---|---------|--------------|-------------------|----------|--------------|
| 1 | Scale per position | D separate scales (per-dimension) | 1 shared scale (isotropic) | **CRITICAL** | Step E |
| 2 | Causal mechanism | Causal mask + output shift (correct) | SOS + self-inclusive causal mask (incorrect Jacobian) | **CRITICAL** | Step A |
| 3 | Scale clamping | None | Asymmetric arctan (alpha_pos=0.1, alpha_neg=2.0) | **CRITICAL** | Step F |
| 4 | Log-det regularization | None | Quadratic penalty on log_det/dof | **HIGH** | Step F |
| 5 | Gaussian noise augmentation | sigma=0.05 | None | **HIGH** | Step D |
| 6 | Learned prior variance | EMA tracked (unused in NVP mode) | Fixed N(0,I) | MEDIUM | N/A (same in NVP mode) |
| 7 | Log-det computation | mean over dims: -xa.mean([1,2]) | sum: (3*s*mask).sum(-1) / dof | MEDIUM | — |
| 8a | Layers per flow block | Multiple (typically 8) | 1 | MEDIUM-HIGH | — |
| 8b | Normalization | Pre-norm | Post-norm | MEDIUM | — |
| 8c | Dropout | None | 0.1 | MEDIUM | — |
| 9 | Positional embeddings | Learned | None | MEDIUM | — |
| 10 | Conditioning | Additive class embed in hidden space | Concatenated atom type in input space | LOW-MEDIUM | Step B |
| 11 | Mixed precision | AMP with bfloat16 + float32 LayerNorm | Full float32 | LOW | — |
| 12 | KV caching | Yes (efficient autoregressive sampling) | No (full forward per atom) | LOW | — |
| 13 | CFG guidance | Yes | No | LOW | N/A |

---

## Key Hypotheses for Phase 3

Based on this analysis, the diagnostic ladder should isolate these effects in order of expected impact:

1. **Causal mask bug (#2):** Our model's Jacobian is not triangular. Fixing this (Step A — adopt Apple's architecture directly) is a prerequisite for meaningful comparison. All results from hyp_002 and hyp_003 used an incorrect NLL — the model was optimizing a biased objective.

2. **Per-dim vs. shared scale (#1 + #3):** This is the most important controlled comparison (Step E). If per-dim scale prevents exploitation without clamping, and shared scale triggers it, this confirms the architectural root cause of our failures.

3. **Noise augmentation (#5):** May interact with scale parameterization — noise smoothing could reduce exploitation incentive regardless of scale type (Step D).

4. **Dropout (#8c):** Should be removed when using Apple's architecture. Flows require deterministic forward/inverse for correct density estimation.

---

## Implications for Existing Results

The causal mask bug (Difference #2) means that hyp_002 and hyp_003 trained with an incorrect NLL objective. The log-det contribution was computed assuming a triangular Jacobian, but the self-inclusive attention makes it non-triangular. This means:

- The reported log-det exploitation in hyp_002 may have been partially driven by the incorrect gradient signal from the biased NLL
- The alpha_pos saturation equilibrium in hyp_003 may not fully reproduce with a correctly autoregressive model
- The 18.3% valid fraction achieved in hyp_003 was obtained despite training with incorrect density estimation

This does NOT invalidate the conclusions about TarFlow architecture — the failures had clear signatures and the remediation attempts were genuine. But it does mean that a properly implemented TarFlow (with correct causal masking) could potentially perform differently. This is exactly what und_001 will test.
