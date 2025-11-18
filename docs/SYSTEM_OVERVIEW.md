# HMPO/GSPO Mono-Repo System Overview

This document explains the current repository architecture, the motivation behind each subsystem, and how to repeat the core experiments. It is intended for both humans and LLM-based agents.

## 1. Repository Layout

| Path | Purpose |
|------|---------|
| `harmonic_mean_policy_optimization.py` | Unified Power Mean + HMPO trainer (uses shared sampling utils and MuonClip). |
| `algorithms/gspo_vectorized.py` | Vendored GSPO implementation used as a baseline and reference. |
| `rl_core/sampling.py` | Vectorized response sampling and log-prob computation shared by HMPO + GSPO. |
| `optimizers/muon_clip.py` | Activation-based MuonClip (qk-clip) wrapper; stabilizes attention scores. |
| `test_harmonic_mean_optimization.py` | Comprehensive tests for configs, ratios, AGM state, and import smoke tests. |
| `test_hmpo_demo.py` | Quick demo/test harness for HMPO. |
| `requirements.txt` | Minimal dependency list (Torch, NumPy, PyTest, Matplotlib). |
| `docs/` | This overview plus research/design notes. |

## 2. Power Mean + HMPO Trainer

- **Motivation**: Explore the arithmetic-geometric-harmonic progression in policy optimization. The trainer allows fixed means, learnable power means, and AGM-adaptive updates.
- **Key design points**:
  - Input validation and Free Energy Principle advantage normalization remain as in earlier revisions.
  - Response sampling/log-prob computation now lives in `rl_core/sampling.py`, ensuring HMPO and GSPO share identical sequence handling.
  - Any optimizer can be supplied; by default we wrap it in MuonClip for attention stability (see §4).

### Repro Steps
1. `python -m pip install -r requirements.txt`
2. `python test_hmpo_demo.py`
3. `pytest -q`

These tests cover configuration validation, importance ratios for all mean types, AGM history population, FEP constraints, and import availability for new modules.

## 3. Vendored GSPO Baseline

- `algorithms/gspo_vectorized.py` is a direct copy of the production-ready GSPO trainer.
- It now imports the shared sampling/log-prob functions and uses MuonClip by default.
- Purpose: provide a baseline for comparisons and ensure HMPO code paths stay aligned with known-good GSPO behavior.

## 4. MuonClip (Activation-Based qk-clip)

- **Motivation**: Fireworks AI’s write-up on Kimi K2 describes MuonClip/qk-clip, which monitors the maximum attention score per batch and rescales `W_q/W_k` only when necessary. This prevents NaNs during trillion-token runs while preserving attention structure.
- **Implementation details**:
  - Registers forward hooks on every module exposing `{q_proj,k_proj}`, `{W_q,W_k}`, `{query,key}`, or `{q,k}`.
  - Reshapes the linear outputs to `(batch, heads, seq, head_dim)` and computes `max(QK^T / sqrt(head_dim))` on the fly.
  - If hooks cannot run, optionally falls back to the previous weight-norm bound.
  - Trainers call `optimizer.step(model=policy_model)` so hooks always see activations.
  - Configuration is centralized in `MuonClipConfig` (thresholds, head-dimension hints, fallback toggles, verbosity).

### Usage Pattern
```python
from optimizers.muon_clip import MuonClip, MuonClipConfig
base_optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
optim = MuonClip(base_optim, MuonClipConfig(max_qk_score=100.0))
# Trainer constructors automatically wrap the optimizer; manual usage is only needed for custom flows.
```

## 5. Repeating Experiments

| Experiment | Steps |
|------------|-------|
| HMPO ratio math & AGM history | `pytest -k harmonic_mean_policy_optimization` |
| Demo run | `python test_hmpo_demo.py` |
| GSPO smoke test | (Optional) `python - <<'PY'` referencing `algorithms.gspo_vectorized` |

For end-to-end RL experiments, replace the dummy reward/model in `test_harmonic_mean_optimization.py` with your task and run the trainer loop; MuonClip will automatically protect attention layers.

## 6. Notes for LLM Agents

- Import paths are stable; prefer `from rl_core.sampling import ...` and `from optimizers.muon_clip import ...`.
- When creating new attention modules, expose `q_proj/k_proj` attributes so MuonClip hooks them without extra configuration.
- Always call `optimizer.step(model=your_model)` if you bypass the trainers.

## 7. External References

- Fireworks AI MuonClip blog post (qk-clip) – rationale for activation-based clipping.
- Kimi K2 Instruct model card – example of MuonClip applied at scale.
- Research prompt and AGM adaptive document for further theoretical context (see other files in `docs/`).

This document should be updated whenever the architecture or stability tools change, ensuring both humans and automated agents can repeat and extend the experiments.


