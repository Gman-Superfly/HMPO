# Harmonic Mean Policy Optimization (HMPO)

[Under construction check back in a week or two]

==============================

A modest exploration of mean types in reinforcement learning policy optimization.

==============================

FYI this is for fun, because GSPO is the convergence point,
"natural balance" between arithmetic and harmonic
so I have no idea how this will play out practically
ie: thought excercise which will have code example


## Background

While working with recent RL algorithms like GRPO and GSPO, I noticed an interesting mathematical pattern:

- **GRPO** uses arithmetic mean of importance ratios
- **GSPO** uses geometric mean (via length normalization)

This made me curious: what about the harmonic mean? The arithmetic-geometric-harmonic mean inequality is a fundamental result in mathematics, and it seemed like the harmonic mean might offer different properties for policy optimization.

## What This Is

This repository contains a simple implementation exploring harmonic mean importance ratios in policy optimization. The main ideas are:

1. **Harmonic Mean Ratios**: Using `n / Σ(1/ratio_i)` instead of arithmetic or geometric means
2. **Power Mean Framework**: Unifying all three approaches as special cases of power means
3. **AGM Connection**: The classical Arithmetic-Geometric Mean algorithm relates these approaches

## Mathematical Foundation

The three mean types form a natural progression:

```
Arithmetic:  (1/n) * Σ x_i           [p = 1]
Geometric:   (Π x_i)^(1/n)           [p = 0] 
Harmonic:    n / Σ(1/x_i)            [p = -1]
```

They satisfy the inequality: Harmonic ≤ Geometric ≤ Arithmetic

Each encodes different risk preferences:
- Arithmetic: Risk-seeking (dominated by largest values)
- Geometric: Risk-neutral (balanced)
- Harmonic: Risk-averse (dominated by smallest values)

## Files

- `algorithms/gspo_vectorized.py` - Vendored GSPO implementation (baseline)
- `rl_core/sampling.py` - Shared vectorized sampling and log-prob utilities
- `HARMONIC_MEAN_RL_EXPLORATION_THOUGHTS.md` - Original research notes

## Quickstart (Windows PowerShell)

```powershell
python -m pip install -r requirements.txt
python .\test_hmpo_demo.py
pytest -q
```

Notes:
- Ensure `eos_token` != `pad_token` in configs.
- HMPO and GSPO now live in this repo; no external repos required.

## Using MuonClip (QK-clip)

You can wrap any optimizer with our MuonClip wrapper to stabilize attention QK scores during training (inspired by Kimi K2’s MuonClip qk-clip).

```python
from optimizers.muon_clip import MuonClip, MuonClipConfig
import torch

optimizer = torch.optim.AdamW(policy_model.parameters(), lr=1e-4)
optimizer = MuonClip(optimizer, MuonClipConfig(max_qk_score=100.0, verbose=True))

loss.backward()
optimizer.step(model=policy_model)  # pass the model so MuonClip can find q/k projections
optimizer.zero_grad()
```

By default it searches for query/key linear layers named `q_proj/k_proj`, `W_q/W_k`, `query/key`, or `q/k` and proportionally rescales them when an estimated max QK score exceeds the threshold.

### Activation-based MuonClip

Our MuonClip wrapper now monitors the **per-batch max QK score** via forward hooks on attention projections. This mirrors the activation-based qk-clip used in Kimi K2 [Fireworks blog](https://fireworks.ai/blog/muonclip). When the live Q·K/√dₖ value exceeds `max_qk_score`, it rescales the offending `W_q/W_k` pair; otherwise it stays inert. If hooks cannot be registered, it falls back to the earlier weight-norm estimate.

See `docs/SYSTEM_OVERVIEW.md` for a full architectural overview and reproducibility notes.

## Key Insight

The geometric mean emerges naturally from the AGM (Arithmetic-Geometric Mean) algorithm:

```python
def agm_step(a, h):
    new_a = (a + h) / 2      # arithmetic mean
    new_h = 2*a*h / (a + h)  # harmonic mean  
    return new_a, new_h

# Iterating this converges to the geometric mean: √(a₀ * h₀)
```

This suggests that GSPO's effectiveness might come from being the natural "balance point" between aggressive (arithmetic) and conservative (harmonic) approaches.

## Potential Applications

The harmonic mean's conservative nature might be useful for:
- Safety-critical AI systems
- Domains where robustness matters more than speed
- Applications requiring consistent, predictable behavior

Though this needs empirical validation.

## Status

This is exploratory research. The implementation demonstrates the mathematical concepts but would need significant work for production use, including:

- Integration with real language models
- Proper response sampling implementation  
- Comprehensive benchmarking against existing methods
- Theoretical analysis of convergence properties

## Limitations

- Only theoretical/toy implementation so far
- No empirical comparison with GRPO/GSPO on real tasks
- Mathematical properties need deeper theoretical analysis
- Computational efficiency not yet optimized



## Related Work

- [GRPO Paper] - Group Relative Policy Optimization Deepseek
- [GSPO Paper] - Group Sequence Policy Optimization  Qwen
- Classical literature on mean inequalities and AGM algorithm

## License

MIT License - Feel free to use, modify, or build upon this work.

---

*A small mathematical curiosity that might (or might not) prove useful for RL practitioners, code is currently being worked on, mean calcs needs attention check back in a week or two* 
