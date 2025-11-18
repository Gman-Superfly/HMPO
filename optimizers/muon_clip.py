import math
import weakref
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim


@dataclass
class MuonClipConfig:
    """Configuration for MuonClip stability wrapper inspired by Muon/qk-clip."""

    max_qk_score: float = 100.0
    # Activation-based monitoring parameters
    activation_based: bool = True
    default_num_heads: int = 1
    head_dim_hint: Optional[int] = None
    head_dim_attr_candidates: Tuple[str, ...] = (
        "head_dim",
        "hidden_size_per_head",
        "head_size",
    )
    num_heads_attr_candidates: Tuple[str, ...] = (
        "num_heads",
        "num_attention_heads",
        "n_heads",
        "num_q_heads",
    )
    query_key_attr_pairs: Tuple[Tuple[str, str], ...] = (
        ("q_proj", "k_proj"),
        ("W_q", "W_k"),
        ("query", "key"),
        ("q", "k"),
    )
    # Weight-norm fallback parameters
    use_weight_norm_fallback: bool = True
    norm_type: str = "fro"  # "fro" | "spectral"
    verbose: bool = False


class _QKMonitor:
    """Forward-hook monitor that records per-batch max QK scores."""

    def __init__(
        self,
        parent: "MuonClip",
        module_name: str,
        module: nn.Module,
        q_linear: nn.Linear,
        k_linear: nn.Linear,
        config: MuonClipConfig,
    ):
        self.parent = parent
        self.module_name = module_name
        self.module = module
        self.q_linear = q_linear
        self.k_linear = k_linear
        self.config = config
        self.last_score: Optional[float] = None
        self._latest_q: Optional[torch.Tensor] = None
        self._latest_k: Optional[torch.Tensor] = None
        self.num_heads = self._infer_num_heads()
        self.head_dim = self._infer_head_dim()
        self.handles = [
            q_linear.register_forward_hook(self._hook_q),
            k_linear.register_forward_hook(self._hook_k),
        ]

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def _infer_num_heads(self) -> int:
        for attr in self.config.num_heads_attr_candidates:
            if hasattr(self.module, attr):
                try:
                    value = int(getattr(self.module, attr))
                    if value > 0:
                        return value
                except Exception:
                    continue
        return max(1, self.config.default_num_heads)

    def _infer_head_dim(self) -> int:
        for attr in self.config.head_dim_attr_candidates:
            if hasattr(self.module, attr):
                try:
                    value = int(getattr(self.module, attr))
                    if value > 0:
                        return value
                except Exception:
                    continue
        if self.config.head_dim_hint:
            return self.config.head_dim_hint
        out_features = self.q_linear.out_features
        if out_features % self.num_heads == 0:
            return out_features // self.num_heads
        return out_features

    def _hook_q(self, _module, _inputs, output):
        if not self.config.activation_based:
            return
        self._latest_q = output.detach()

    def _hook_k(self, _module, _inputs, output):
        if not self.config.activation_based:
            return
        self._latest_k = output.detach()
        self._compute_activation()

    def _compute_activation(self):
        if self._latest_q is None or self._latest_k is None:
            return
        q = self._latest_q
        k = self._latest_k
        self._latest_q = None
        self._latest_k = None

        if q.dim() == 2:
            q = q.unsqueeze(0)
        if k.dim() == 2:
            k = k.unsqueeze(0)
        if q.dim() != 3 or k.dim() != 3:
            return

        batch, q_seq, hidden = q.shape
        _, k_seq, hidden_k = k.shape
        if hidden != hidden_k or self.head_dim <= 0 or hidden % self.head_dim != 0:
            return
        heads = hidden // self.head_dim
        q = q.view(batch, q_seq, heads, self.head_dim).permute(0, 2, 1, 3).reshape(batch * heads, q_seq, self.head_dim)
        k = k.view(batch, k_seq, heads, self.head_dim).permute(0, 2, 1, 3).reshape(batch * heads, k_seq, self.head_dim)
        scores = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.head_dim)
        max_score = float(scores.max().item())
        if self.last_score is None or max_score > self.last_score:
            self.last_score = max_score
        self.parent._record_activation(self, max_score)

    def consume_last_score(self) -> Optional[float]:
        score = self.last_score
        self.last_score = None
        return score

    def scale_weights(self, scale: float):
        self.q_linear.weight.mul_(scale)
        self.k_linear.weight.mul_(scale)


class MuonClip(optim.Optimizer):
    """Optimizer wrapper that applies MuonClip's qk-clip after each step."""

    def __init__(
        self,
        base_optimizer: optim.Optimizer,
        config: Optional[MuonClipConfig] = None,
        model: Optional[nn.Module] = None,
    ):
        if not isinstance(base_optimizer, optim.Optimizer):
            raise TypeError("base_optimizer must be a torch.optim.Optimizer")
        self.base_optimizer = base_optimizer
        self.config = config or MuonClipConfig()
        self.param_groups = self.base_optimizer.param_groups
        self.state = self.base_optimizer.state
        self.last_clip_stats: Dict[str, Any] = {}
        self._model_ref: Optional[weakref.ReferenceType[nn.Module]] = None
        self._monitors: List[_QKMonitor] = []
        self._hooks_registered = False
        if model is not None:
            self.attach_model(model)

    def __getattr__(self, name: str):
        if name in (
            "base_optimizer",
            "config",
            "param_groups",
            "state",
            "last_clip_stats",
            "_model_ref",
            "_monitors",
            "_hooks_registered",
        ):
            return super().__getattribute__(name)
        return getattr(self.base_optimizer, name)

    def attach_model(self, model: nn.Module):
        self._model_ref = weakref.ref(model)
        self._register_hooks(model)

    @torch.no_grad()
    def step(self, closure=None, model: Optional[nn.Module] = None):
        loss = self.base_optimizer.step(closure) if closure is not None else self.base_optimizer.step()
        target_model = model or (self._model_ref() if self._model_ref else None)
        if target_model is not None:
            self._ensure_hooks(target_model)
            self._apply_qk_clip(target_model)
        return loss

    def _ensure_hooks(self, model: nn.Module):
        if not self._hooks_registered:
            self.attach_model(model)

    def _register_hooks(self, model: nn.Module):
        self._hooks_registered = True
        self._monitors = []
        for module_name, module in model.named_modules():
            for q_attr, k_attr in self.config.query_key_attr_pairs:
                q_lin = getattr(module, q_attr, None)
                k_lin = getattr(module, k_attr, None)
                if isinstance(q_lin, nn.Linear) and isinstance(k_lin, nn.Linear):
                    monitor = _QKMonitor(self, module_name, module, q_lin, k_lin, self.config)
                    self._monitors.append(monitor)
                    break

    def close(self):
        for monitor in self._monitors:
            monitor.close()
        self._monitors = []
        self._hooks_registered = False

    def _record_activation(self, monitor: _QKMonitor, score: float):
        # Placeholder for future analytics; nothing required here beyond monitor.last_score.
        pass

    @torch.no_grad()
    def _apply_qk_clip(self, model: nn.Module) -> None:
        cfg = self.config
        clipped_layers: List[Tuple[str, float, float, float]] = []

        used_activation = False
        if cfg.activation_based and self._monitors:
            used_activation = True
            for monitor in self._monitors:
                score = monitor.consume_last_score()
                if score is None:
                    continue
                if score > cfg.max_qk_score:
                    scale = math.sqrt(cfg.max_qk_score / (score + 1e-12))
                    monitor.scale_weights(scale)
                    clipped_layers.append((monitor.module_name, float(score), float(cfg.max_qk_score), float(scale)))

        if (not clipped_layers) and cfg.use_weight_norm_fallback:
            for module_name, module in model.named_modules():
                for q_attr, k_attr in cfg.query_key_attr_pairs:
                    q_lin = getattr(module, q_attr, None)
                    k_lin = getattr(module, k_attr, None)
                    if isinstance(q_lin, nn.Linear) and isinstance(k_lin, nn.Linear):
                        q_norm = self._matrix_norm(q_lin.weight, cfg.norm_type)
                        k_norm = self._matrix_norm(k_lin.weight, cfg.norm_type)
                        head_dim = cfg.head_dim_hint or min(q_lin.weight.shape)
                        sqrt_dk = float(head_dim) ** 0.5 if head_dim and head_dim > 0 else 1.0
                        est_max_qk = (q_norm * k_norm) / sqrt_dk
                        if est_max_qk > cfg.max_qk_score:
                            scale = math.sqrt(cfg.max_qk_score / (est_max_qk + 1e-12))
                            q_lin.weight.mul_(scale)
                            k_lin.weight.mul_(scale)
                            clipped_layers.append((f"{module_name}.{q_attr}/{k_attr}", float(est_max_qk), float(cfg.max_qk_score), float(scale)))
                        break

        self.last_clip_stats = {
            "num_layers_clipped": len(clipped_layers),
            "layers": clipped_layers,
            "activation_based": used_activation and bool(clipped_layers),
            "max_qk_score": cfg.max_qk_score,
        }
        if cfg.verbose and clipped_layers:
            names = ", ".join(f"{name}[scale={scale:.3f}]" for name, _, _, scale in clipped_layers)
            print(f"[MuonClip] Clipped {len(clipped_layers)} layer pairs: {names}")

    @staticmethod
    def _matrix_norm(w: torch.Tensor, norm_type: str) -> float:
        if norm_type == "spectral":
            try:
                return float(torch.linalg.svdvals(w).max().item())
            except Exception:
                return float(torch.linalg.matrix_norm(w, 2).item())
        return float(torch.linalg.matrix_norm(w, "fro").item())


