## Diagnostic Report — hyp_010: TarFlow Apple Multi-Molecule

### Baseline Failure Analysis

Phase 1 (ethanol T=9): VF=95%. PASSED.

Phase 2 (ethanol T=21 with padding mask): VF=33.2%. FAILED.

The 62pp gap between T=9 (95%) and T=21 (33%) was large enough to be a clear implementation bug rather than a capacity issue. Both configs used identical model weights initialized from scratch, same training hyperparameters, and same molecule. Only the `seq_length` (9 vs 21) and `use_padding_mask` (False vs True) differed.

**What fraction fail?** ~67% of T=21 samples failed the valid fraction criterion. The generated configurations had incorrect interatomic distances, suggesting the autoregressive chain was corrupted rather than simply under-trained.

**Diagnostic approach:** Load the T=21 final checkpoint, sample with and without padding mask, inspect real vs. padding position statistics. Then trace through the forward and reverse pass code to identify where padding noise enters the generation chain.

### Root Cause

Two bugs were identified, both in `src/train_phase3.py`:

**Bug 1 — Sampling noise at padding positions (partial contributor, ~20pp of gap):**

In `TarFlow1DMol.sample()`:
```python
z = torch.randn(n, seq_length, in_channels, device=device) * temp
```
This initializes the latent `z` with Gaussian noise for ALL T=21 positions, including the 12 padding positions (indices 9-20 for ethanol).

The forward pass correctly produces `z_pad = 0` by zeroing `xa, xb` for padding positions (via `mask_perm`). But the reverse pass receives full Gaussian noise at padding positions.

In PermutationFlip blocks, the sequence is reversed: position `i` becomes position `T-1-i`. Padding atoms originally at positions 9-20 appear at positions 0-11 in permuted space. Since the autoregressive generation chain processes position 0, 1, 2, ... in order, the Gaussian noise at the first 12 positions provides corrupted context for generating real atoms starting at position 12.

With Fix 1 only (zero padding in `z` before reverse + re-zero between blocks): VF improved from 33% to 47%.

**Bug 2 — Padding key masking in attention (dominant contributor, ~47pp of gap):**

In `MetaBlockWithCond.forward()`, the attention mask was:
```python
pad_key_mask = mask_perm.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
# 0 where padding (masked), 1 where real
attn_mask_combined = self.base.attn_mask * pad_key_mask
```

This combined the causal (lower triangular) mask with a padding KEY mask that zeroed out all key positions belonging to padding atoms.

In PermutationFlip blocks, padding atoms appear first (positions 0-11). Position 11 (the last padding token, immediately before the first real atom at position 12) has:
- Its own causal mask: rows 0-11 visible
- Its padding key mask: positions 0-11 all masked (they are padding)
- Result: the transformer at position 11 sees NO valid keys — all keys are masked

This produces near-zero affine transform parameters for position 12 (the first real atom). With no conditioning signal, the first real atom generation is degenerate, which cascades through the autoregressive chain.

With both fixes: VF = 93.6% on T=21 ethanol. Gap = 1.4pp vs T=9 (95%). Phase 2 PASSED.

**Why Fix 2 was necessary separately from Fix 1:**
Fix 1 (zeroing padding latent) ensures no padding NOISE enters the chain. But Fix 2 addresses a different issue: even if padding positions have `x_perm=0` (zeros), the attention mask was ALSO suppressing the atom type embeddings and positional embeddings at those positions from being used as context. Fix 2 removes this suppression. The padding tokens, even with zero coordinate values, carry useful conditioning information via their atom type embeddings (type 0 = "padding") and positional encodings.

### Priority Order Assessment

| Phase | Applicable? | Rationale |
|-------|-------------|-----------|
| SANITY | Yes | The T=9 → T=21 gap is a clear implementation bug (62pp), not a capacity issue. Fixing the padding handling before any other angle. This was Phase 1+2 of the task spec. |
| KNOWN HEURISTICS | Yes (conditional) | If Phase 3 multi-molecule fails after SANITY, heuristics (lr schedule, ldr sweep) are the next step. |
| SCALE | Yes (conditional) | If heuristics fail, scale channels and training steps. |

### Proposed Angles (preliminary)

Phase 3 SANITY: Multi-molecule run, T=21, all 8 molecules, 20k steps. Uses the fixed `src/train_phase3.py`. Success: VF > 50% ethanol AND mean VF > 40%.

If SANITY passes → done (may try HEURISTICS sweep to further improve).
If SANITY fails but VF > 20% at step 5k → try HEURISTICS sweep (lr, ldr).
If SANITY fails and VF < 20% → investigate further (possible multi-molecule issues).
