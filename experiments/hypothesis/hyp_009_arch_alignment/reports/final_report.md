## Final Experiment Report — hyp_009: Architecture Alignment
**Status:** FAILURE
**Branch:** `exp/hyp_009`
**Commits:** [`6e4f659` — meta: initialize experiment branch and state file], [`9a8a450` — code: pre-norm + layers_per_block implementation], [`e1cb140` — config: pre-run snapshot for phase1 validation run], [`cee25bd` — config: diagnostic investigation configs], [`1a4515b` — config: contraction-only diagnostic]

### Experimental Outcome
Phase 1 gate FAILED. Pre-norm + layers_per_block=2 on model.py produced 14% VF on ethanol T=9 — WORSE than the post-norm baseline (~39%). Multiple diagnostic configurations were tried but none approached the 90% target.

This is the 4th consecutive experiment (hyp_006 through hyp_009) attempting to incrementally patch model.py to match tarflow_apple.py's performance. Each experiment identified and fixed a "root cause" but VF never exceeded 55.8% (hyp_007 multi-mol) or 39.2% (hyp_008 single-mol ethanol).

The conclusion: the 13+ architectural differences between model.py and tarflow_apple.py cannot be isolated and fixed incrementally. The proven architecture (tarflow_apple.py + TarFlow1DMol, achieving 96-98% VF in und_001) should be used directly.

### Project Context
This failure confirms the need to abandon incremental patching and use tarflow_apple.py directly for multi-molecule training. The model.py codebase has accumulated complexity from 7 experiments (hyp_002 through hyp_009) without closing the VF gap. TarFlow1DMol from und_001 already achieves 98.2% mean VF per-molecule. Next experiment (hyp_010) will use TarFlow1DMol for multi-molecule training.

### Story Validation
CONFLICT — The incremental approach to architecture alignment has failed. The research story's Experiment Plan should be updated to reflect the pivot to direct use of tarflow_apple.py for multi-molecule training.

### Open Questions
None — the path forward is clear: use the proven architecture directly.
