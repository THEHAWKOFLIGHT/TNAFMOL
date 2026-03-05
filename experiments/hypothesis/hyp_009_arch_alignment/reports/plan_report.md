## Plan Sub-report — hyp_009: Architectural Alignment (Pre-Norm + Layers Per Block)

**Status:** READY_TO_START

### Understanding

hyp_008 established the TRUE root cause of the 56pp VF gap between model.py (~39% VF) and tarflow_apple.py (~96% VF):
1. **Post-norm vs pre-norm**: model.py applies LayerNorm AFTER residual add. Apple applies it BEFORE the sublayer (pre-norm).
2. **Layers per flow block**: model.py has 1 attention+FFN per TarFlowBlock. Apple has `num_layers=2` per MetaBlock.
3. **Dropout**: model.py uses dropout=0.1. Apple uses dropout=0.0.

This experiment adds `use_pre_norm` (bool) and `layers_per_block` (int) parameters to TarFlowBlock and TarFlow, and runs a three-phase validation.

Evidence for these being the true root causes:
- und_001 Phase 4 showed per_dim_scale difference is <1pp (96.2% vs 95.3%)
- tarflow_apple.py MetaBlock uses pre-norm explicitly via `self.norm` in Attention and MLP classes (lines 55, 107)
- tarflow_apple.py MetaBlock uses `nn.ModuleList([AttentionBlock(...) for _ in range(num_layers)])` with num_layers=2 (default)
- These are structural capacity differences that would compound across 4 flow blocks

### Execution Plan

**Phase 1: Implement + Unit Test + Validate (GATE)**
1. Modify `src/model.py`:
   - Add `use_pre_norm: bool = False`, `layers_per_block: int = 1` to TarFlowBlock.__init__
   - Replace single attention+FFN attributes with nn.ModuleList of layers
   - Add pre-norm path in _run_transformer and _run_transformer_output_shift
   - Post-norm path must be mathematically identical to original (backward compat)
   - Add `final_norm` LayerNorm when use_pre_norm=True
   - Propagate use_pre_norm + layers_per_block to TarFlow

2. Modify `src/train.py`:
   - Add `use_pre_norm: False`, `layers_per_block: 1` to DEFAULT_CONFIG
   - Pass to TarFlow constructor

3. Unit tests (mandatory before any training):
   - Test 1: Post-norm path identical to old code
   - Test 2: Pre-norm produces different output
   - Test 3: Forward-inverse consistency (use_pre_norm=True)
   - Test 4: layers_per_block=2 increases parameter count correctly
   - Test 5: Forward-inverse with layers_per_block=2, use_pre_norm=True
   - Test 6: Jacobian triangularity preserved

4. Phase 1 validation run (cuda:8, 5k steps, ethanol T=9):
   - Config: use_pre_norm=True, layers_per_block=2, dropout=0.0, n_blocks=4, d_model=256
   - Gate: VF >= 90% → proceed to Phase 2; VF 70-90% → investigate; VF < 70% → debug

**Phase 2: Padding Re-Validation**
- Run A: ethanol T=9 (max_atoms=9), 5k steps, cuda:8
- Run B: ethanol T=21 (max_atoms=21), 5k steps, cuda:9
- Both in parallel, same config otherwise
- Success: |VF(T=9) - VF(T=21)| < 10pp AND both >= 85%

**Phase 3: Multi-Molecule OPTIMIZE (SANITY angle)**
- Config: use_pre_norm=True, layers_per_block=2, dropout=0.0, all 8 molecules, max_atoms=21
- 20k steps via sbatch (production run)
- Promising criterion: VF > 50% on ethanol AND mean VF > 40% across 8 molecules
- If promising: HEURISTICS sweep (lr x ldr x n_blocks)
- If not promising: SCALE (d_model=384, n_blocks=8, 50k steps)

### Proposed Milestones

- Milestone 1: Implementation + unit tests pass
- Milestone 2: Phase 1 gate (VF >= 90% on ethanol T=9)
- Milestone 3: Phase 2 padding validation
- Milestone 4: Phase 3 multi-molecule SANITY

### Key Implementation Decisions

**Pre-norm vs post-norm mathematical equivalence check:**
- Post-norm: `h = LayerNorm(h + dropout(attn(h)))`; `h = LayerNorm(h + ffn(h))`
- Pre-norm: `h = h + dropout(attn(LayerNorm(h)))`; `h = h + ffn(LayerNorm(h))`
- When layers_per_block=1 and use_pre_norm=False, the new code must produce IDENTICAL output to the original.

**Backward compat:**
- Default use_pre_norm=False, layers_per_block=1 → identical to existing behavior
- Only when use_pre_norm=True is the architecture changed

**Dropout:**
- Apple uses dropout=0.0 everywhere in attention and FFN
- We keep the same dropout parameter but set it to 0.0 in Phase 1+ configs

**Autoregressive structure:**
- Pre-norm does NOT change the causal masking logic
- Only the order of norm/residual within the transformer sublayers changes
- The output shift mechanism is unaffected — autoregressive correctness preserved

### Questions / Concerns

None. The architectural changes are well-defined, and the backward-compat test gives confidence the refactoring is correct.
