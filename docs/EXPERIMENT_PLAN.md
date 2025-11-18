# HMPO vs. GSPO Experimental Plan

## Goals
1. **Stability under long rollouts**: quantify how often GSPO and HMPO encounter exploding attention scores or NaNs, with and without MuonClip.
2. **Reward/advantage behavior**: compare average reward, advantage variance, and clipped-ratio fractions to understand each mean’s risk profile.
3. **Learning dynamics**: measure convergence speed and sample efficiency (episodes to reach target reward) on representative tasks.
4. **AGM telemetry**: record arithmetic/harmonic histories and AGM convergence rates so the external adaptive controllers have real data.

## Metrics
| Category | Metric | Notes |
|----------|--------|-------|
| Stability | Max Q·K/√d per batch | Captured via MuonClip hooks; track percent of steps that required clipping. |
|           | NaN/Inf occurrences | Count optimizer steps skipped due to numerical errors. |
| Reward    | Mean episode reward | Rolling average ± std-dev across seeds. |
|           | Advantage variance | Tests Free Energy normalization + mean-type effect. |
| Ratios    | Mean/clipped ratio stats | Compare aggressiveness vs conservatism. |
| Learning  | Steps to reach target reward | Sample efficiency proxy. |
| AGM       | |Arithmetic - Harmonic| gap | Feed to AGM controllers, also reflects convergence health. |

## Experimental Variants
1. **Baseline GSPO**: GSPO trainer with MuonClip off/on.
2. **HMPO (harmonic)**: HMPO trainer with MuonClip on (default). Optionally test without MuonClip to show stability delta.
3. **Power Mean sweep**: p ∈ {1, 0, -1, learnable} on a shared reward function.
4. **AGM Adaptive**: if time permits, run AGM-adaptive mean selection to produce telemetry for the AGM repo.

## Tasks & Setup (to be detailed in next steps)
- Select a small language-model or sequence task (e.g., summarization or synthetic reward) that both trainers can run quickly.
- Define a deterministic reward function for reproducibility.
- Use consistent seeds and logging (Weights & Biases or simple CSV) to capture all metrics above.

## Next Steps
1. Implement logging hooks in HMPO/GSPO to emit the metrics listed.
2. Build a script (or notebook) that runs the variants and stores results per seed.
3. Share telemetry with the AGM repo for controller development.


