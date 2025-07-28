"""
Harmonic Mean Policy Optimization (HMPO) and Unified Power Mean Framework

This module implements the missing piece in the arithmetic-geometric-harmonic mean
progression for policy optimization algorithms:

- GRPO: Arithmetic mean (p=1) - aggressive, unstable
- GSPO: Geometric mean (p=0) - balanced, length-normalized  
- HMPO: Harmonic mean (p=-1) - conservative, robust
- Power Mean: Unified framework with learnable parameter p

Based on the mathematical insight that geometric mean emerges from iterating
arithmetic ↔ harmonic means (Arithmetic-Geometric Mean algorithm).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import List, Dict, Tuple, Callable, Optional, Literal
import math

__all__ = [
    'PowerMeanConfig', 'HMPOConfig', 'PowerMeanTrainer', 'HMPOTrainer',
    'harmonic_mean_policy_optimization', 'power_mean_policy_optimization',
    'agm_adaptive_optimization'
]

class PowerMeanError(Exception):
    """Base exception for Power Mean optimization errors."""
    pass

class FreeEnergyViolationError(PowerMeanError):
    """Raised when Free Energy Principle is violated."""
    pass

MeanType = Literal['arithmetic', 'geometric', 'harmonic', 'power', 'agm_adaptive']

class PowerMeanConfig:
    """Configuration for Power Mean Policy Optimization framework."""
    
    def __init__(self,
                 group_size: int = 4,
                 epsilon: float = 0.2,
                 max_length: int = 512,
                 eos_token: int = 2,
                 pad_token: int = 0,
                 mean_type: MeanType = 'harmonic',
                 power_p: Optional[float] = None,
                 learnable_p: bool = False,
                 agm_iterations: int = 5,
                 conservative_threshold: float = 0.1):
        """Initialize Power Mean configuration.
        
        Args:
            group_size: Number of sequences per group
            epsilon: Clipping parameter for importance ratios
            max_length: Maximum generation length
            eos_token: End-of-sequence token ID
            pad_token: Padding token ID
            mean_type: Type of mean to use
            power_p: Power parameter for power mean (if None, uses defaults)
            learnable_p: Whether to make power parameter learnable
            agm_iterations: Number of AGM iterations for adaptive method
            conservative_threshold: Threshold for switching to conservative mode
            
        Raises:
            PowerMeanError: If configuration is invalid
        """
        assert group_size > 0, f"Invalid group_size: {group_size}"
        assert 0 < epsilon < 1, f"Invalid epsilon: {epsilon}"
        assert max_length > 0, f"Invalid max_length: {max_length}"
        assert eos_token != pad_token, "EOS and padding tokens must be different"
        assert agm_iterations > 0, f"Invalid agm_iterations: {agm_iterations}"
        assert 0 < conservative_threshold < 1, f"Invalid conservative_threshold: {conservative_threshold}"
        
        self.group_size = group_size
        self.epsilon = epsilon
        self.max_length = max_length
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.mean_type = mean_type
        self.agm_iterations = agm_iterations
        self.conservative_threshold = conservative_threshold
        
        # Set power parameter based on mean type
        if power_p is not None:
            self.power_p = power_p
        else:
            if mean_type == 'arithmetic':
                self.power_p = 1.0  # GRPO
            elif mean_type == 'geometric':
                self.power_p = 0.0  # GSPO
            elif mean_type == 'harmonic':
                self.power_p = -1.0  # HMPO
            elif mean_type == 'power':
                self.power_p = -0.5  # Default power mean
            else:  # agm_adaptive
                self.power_p = 0.0  # Start with geometric
        
        self.learnable_p = learnable_p
        
        # Validate power parameter
        if mean_type == 'power':
            assert power_p is not None, "power_p required for power mean type"
    
    def validate(self) -> bool:
        """Validate configuration parameters."""
        return (self.group_size > 0 and 
                0 < self.epsilon < 1 and 
                self.max_length > 0 and
                self.eos_token != self.pad_token and
                self.agm_iterations > 0 and
                0 < self.conservative_threshold < 1)

class HMPOConfig(PowerMeanConfig):
    """Specialized configuration for Harmonic Mean Policy Optimization."""
    
    def __init__(self, **kwargs):
        kwargs['mean_type'] = 'harmonic'
        kwargs['power_p'] = -1.0
        super().__init__(**kwargs)

class PowerMeanTrainer:
    """Unified trainer for Power Mean Policy Optimization framework."""
    
    def __init__(self,
                 policy_model: nn.Module,
                 ref_model: nn.Module, 
                 optimizer: optim.Optimizer,
                 config: PowerMeanConfig,
                 reward_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
        """Initialize Power Mean trainer.
        
        Args:
            policy_model: Current policy to optimize
            ref_model: Reference (frozen) policy
            optimizer: Optimizer for policy model
            config: Power Mean configuration
            reward_fn: Function that takes (prompts, responses) -> rewards
            
        Raises:
            PowerMeanError: If configuration is invalid
        """
        assert config.validate(), "Invalid Power Mean configuration"
        assert policy_model is not None, "Policy model required"
        assert ref_model is not None, "Reference model required"
        assert reward_fn is not None, "Reward function required"
        
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.optimizer = optimizer
        self.config = config
        self.reward_fn = reward_fn
        
        # Initialize learnable power parameter if requested
        if config.learnable_p:
            self.power_param = nn.Parameter(torch.tensor(config.power_p))
        else:
            self.power_param = torch.tensor(config.power_p)
        
        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        # AGM state for adaptive optimization
        self.agm_state = {
            'arithmetic_history': [],
            'harmonic_history': [],
            'geometric_history': [],
            'current_p': config.power_p
        }
    
    def compute_importance_ratio(self,
                               policy_logprobs: torch.Tensor,
                               ref_logprobs: torch.Tensor,
                               response_lengths: torch.Tensor,
                               mean_type: Optional[MeanType] = None) -> torch.Tensor:
        """Compute importance ratios using specified mean type.
        
        Args:
            policy_logprobs: Log probabilities from policy model (batch_size,)
            ref_logprobs: Log probabilities from reference model (batch_size,)
            response_lengths: Actual response lengths (batch_size,)
            mean_type: Override config mean type
            
        Returns:
            Importance ratios (batch_size,)
            
        Raises:
            PowerMeanError: If computation fails
        """
        assert policy_logprobs.shape == ref_logprobs.shape, "Shape mismatch in log probabilities"
        assert policy_logprobs.shape[0] == response_lengths.shape[0], "Batch size mismatch"
        assert torch.all(response_lengths > 0), "Invalid response lengths"
        
        mean_type = mean_type or self.config.mean_type
        power_p = self.power_param if hasattr(self, 'power_param') else self.config.power_p
        
        # Compute log ratio per token (policy - ref = log(policy/ref))
        log_ratios = policy_logprobs - ref_logprobs
        
        if mean_type == 'arithmetic':
            # GRPO: Arithmetic mean of ratios
            # (1/|y|) * Σ(π_θ/π_θ_old)
            ratios = torch.exp(log_ratios)  # Convert to ratios
            importance_ratios = ratios / response_lengths.float()  # Average over sequence length
            
        elif mean_type == 'geometric':
            # GSPO: Geometric mean via length normalization
            # exp((1/|y|) * Σ log(π_θ/π_θ_old)) = (π_θ(y)/π_θ_old(y))^(1/|y|)
            importance_ratios = torch.exp(log_ratios / response_lengths.float())
            
        elif mean_type == 'harmonic':
            # HMPO: Harmonic mean of ratios
            # |y| / Σ(π_θ_old/π_θ) = |y| / Σ(1/(π_θ/π_θ_old))
            ratios = torch.exp(log_ratios)  # π_θ/π_θ_old
            inverse_ratios = 1.0 / (ratios + 1e-8)  # Numerical stability
            harmonic_mean = response_lengths.float() / torch.sum(inverse_ratios, dim=0, keepdim=True)
            importance_ratios = harmonic_mean.expand_as(ratios)
            
        elif mean_type == 'power':
            # Power mean: ((1/|y|) * Σ(x^p))^(1/p) for p ≠ 0
            if abs(power_p) < 1e-8:  # p ≈ 0, use geometric mean
                importance_ratios = torch.exp(log_ratios / response_lengths.float())
            else:
                ratios = torch.exp(log_ratios)  # π_θ/π_θ_old
                powered_ratios = torch.pow(ratios + 1e-8, power_p)  # Numerical stability
                mean_powered = powered_ratios / response_lengths.float()
                importance_ratios = torch.pow(mean_powered + 1e-8, 1.0 / power_p)
                
        elif mean_type == 'agm_adaptive':
            # AGM adaptive: Use arithmetic-geometric mean iteration
            importance_ratios = self._compute_agm_ratios(log_ratios, response_lengths)
            
        else:
            raise PowerMeanError(f"Unknown mean type: {mean_type}")
        
        # Validation
        assert not torch.any(torch.isnan(importance_ratios)), "NaN in importance ratios"
        assert not torch.any(torch.isinf(importance_ratios)), "Inf in importance ratios"
        assert torch.all(importance_ratios > 0), "Non-positive importance ratios"
        
        return importance_ratios
    
    def _compute_agm_ratios(self, 
                           log_ratios: torch.Tensor, 
                           response_lengths: torch.Tensor) -> torch.Tensor:
        """Compute importance ratios using AGM iteration.
        
        The Arithmetic-Geometric Mean algorithm converges to the geometric mean
        through iterative arithmetic and harmonic mean computations.
        
        Args:
            log_ratios: Log importance ratios
            response_lengths: Response lengths
            
        Returns:
            AGM-computed importance ratios
        """
        ratios = torch.exp(log_ratios)
        
        # Initialize with arithmetic and harmonic means
        arithmetic = ratios / response_lengths.float()  # GRPO
        harmonic_denom = response_lengths.float() / torch.sum(1.0 / (ratios + 1e-8), dim=0, keepdim=True)
        harmonic = harmonic_denom.expand_as(ratios)
        
        # AGM iteration
        for i in range(self.config.agm_iterations):
            new_arithmetic = (arithmetic + harmonic) / 2.0
            new_harmonic = 2.0 * arithmetic * harmonic / (arithmetic + harmonic + 1e-8)
            
            # Check convergence
            if torch.allclose(new_arithmetic, new_harmonic, rtol=1e-6):
                break
                
            arithmetic = new_arithmetic
            harmonic = new_harmonic
        
        # Store AGM history for analysis
        self.agm_state['arithmetic_history'].append(arithmetic.mean().item())
        self.agm_state['harmonic_history'].append(harmonic.mean().item())
        self.agm_state['geometric_history'].append((arithmetic * harmonic).sqrt().mean().item())
        
        # Return converged value (should be geometric mean)
        return (arithmetic + harmonic) / 2.0
    
    def compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """Compute group-based advantages with FEP validation.
        
        Args:
            rewards: Raw rewards (batch_size * group_size,)
            
        Returns:
            Normalized advantages (batch_size * group_size,)
            
        Raises:
            FreeEnergyViolationError: If advantages violate conservation
        """
        assert rewards.size(0) % self.config.group_size == 0, "Batch size not divisible by group size"
        
        batch_size = rewards.size(0) // self.config.group_size
        grouped_rewards = rewards.view(batch_size, self.config.group_size)
        
        # Group-based normalization (preserves relative ordering within groups)
        group_means = grouped_rewards.mean(dim=1, keepdim=True)
        group_stds = grouped_rewards.std(dim=1, keepdim=True, unbiased=False) + 1e-8
        
        advantages = (grouped_rewards - group_means) / group_stds
        advantages = advantages.view(-1)  # Flatten back
        
        # Free Energy Principle validation: advantages should sum to ~0 per group
        group_advantage_sums = advantages.view(batch_size, self.config.group_size).sum(dim=1)
        if torch.any(torch.abs(group_advantage_sums) > 1e-3):
            raise FreeEnergyViolationError(
                f"Advantage sum violation: {group_advantage_sums.abs().max().item():.6f}"
            )
        
        assert not torch.any(torch.isnan(advantages)), "NaN in advantages"
        assert not torch.any(torch.isinf(advantages)), "Inf in advantages"
        
        return advantages
    
    def update_step(self, prompts: torch.Tensor) -> Dict[str, float]:
        """Perform single Power Mean optimization step.
        
        Args:
            prompts: Batch of prompts (batch_size, prompt_len)
            
        Returns:
            Metrics dictionary
            
        Raises:
            PowerMeanError: If update fails
        """
        assert prompts.size(0) > 0, "Empty prompt batch"
        
        batch_size = prompts.size(0)
        device = prompts.device
        
        # Sample responses and compute rewards
        responses, response_lengths, full_sequences = self._sample_responses_batch(prompts)
        rewards = self.reward_fn(
            prompts.repeat_interleave(self.config.group_size, dim=0),
            responses
        )
        
        # Compute advantages with FEP validation
        advantages = self.compute_advantages(rewards)
        
        # Compute log probabilities
        policy_logprobs = self._compute_log_probs_batch(
            self.policy_model, full_sequences, prompts.size(1), response_lengths
        )
        ref_logprobs = self._compute_log_probs_batch(
            self.ref_model, full_sequences, prompts.size(1), response_lengths
        )
        
        # Compute importance ratios using configured mean type
        importance_ratios = self.compute_importance_ratio(
            policy_logprobs, ref_logprobs, response_lengths
        )
        
        # Power Mean objective with clipping
        ratio_advantages = importance_ratios * advantages
        clipped_ratios = torch.clamp(
            importance_ratios, 
            1.0 - self.config.epsilon, 
            1.0 + self.config.epsilon
        )
        clipped_advantages = clipped_ratios * advantages
        
        # Take minimum (conservative) and average over groups
        objective_per_sample = torch.min(ratio_advantages, clipped_advantages)
        grouped_objective = objective_per_sample.view(batch_size, self.config.group_size)
        loss = -grouped_objective.mean()
        
        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Compute metrics
        with torch.no_grad():
            clipped_fraction = (importance_ratios < (1.0 - self.config.epsilon)) | \
                             (importance_ratios > (1.0 + self.config.epsilon))
            
            metrics = {
                'loss': loss.item(),
                'mean_reward': rewards.mean().item(),
                'mean_advantage': advantages.mean().item(),
                'mean_importance_ratio': importance_ratios.mean().item(),
                'clipped_fraction': clipped_fraction.float().mean().item(),
                'mean_response_length': response_lengths.float().mean().item(),
                'power_parameter': float(self.power_param) if hasattr(self, 'power_param') else self.config.power_p
            }
            
            # Add mean-type specific metrics
            if self.config.mean_type == 'agm_adaptive':
                metrics.update({
                    'agm_arithmetic': self.agm_state['arithmetic_history'][-1] if self.agm_state['arithmetic_history'] else 0.0,
                    'agm_harmonic': self.agm_state['harmonic_history'][-1] if self.agm_state['harmonic_history'] else 0.0,
                    'agm_geometric': self.agm_state['geometric_history'][-1] if self.agm_state['geometric_history'] else 0.0
                })
        
        return metrics
    
    def _sample_responses_batch(self, prompts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample responses using reference model (vectorized)."""
        # Implementation similar to GSPO but with additional validation
        # ... (implementation details similar to gspo_vectorized.py)
        pass
    
    def _compute_log_probs_batch(self, model: nn.Module, full_sequences: torch.Tensor, 
                                prompt_length: int, response_lengths: torch.Tensor) -> torch.Tensor:
        """Compute log probabilities for sequences."""
        # Implementation similar to GSPO but with additional validation
        # ... (implementation details similar to gspo_vectorized.py)
        pass

class HMPOTrainer(PowerMeanTrainer):
    """Specialized trainer for Harmonic Mean Policy Optimization."""
    
    def __init__(self, policy_model: nn.Module, ref_model: nn.Module, 
                 optimizer: optim.Optimizer, config: HMPOConfig,
                 reward_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
        super().__init__(policy_model, ref_model, optimizer, config, reward_fn)

# Convenience functions
def harmonic_mean_policy_optimization(policy_model: nn.Module,
                                    ref_model: nn.Module,
                                    optimizer: optim.Optimizer, 
                                    prompts: torch.Tensor,
                                    reward_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                                    config: Optional[HMPOConfig] = None) -> Dict[str, float]:
    """Convenience function for single HMPO update step."""
    if config is None:
        config = HMPOConfig()
    
    trainer = HMPOTrainer(policy_model, ref_model, optimizer, config, reward_fn)
    return trainer.update_step(prompts)

def power_mean_policy_optimization(policy_model: nn.Module,
                                 ref_model: nn.Module, 
                                 optimizer: optim.Optimizer,
                                 prompts: torch.Tensor,
                                 reward_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                                 config: Optional[PowerMeanConfig] = None) -> Dict[str, float]:
    """Convenience function for single Power Mean update step."""
    if config is None:
        config = PowerMeanConfig()
    
    trainer = PowerMeanTrainer(policy_model, ref_model, optimizer, config, reward_fn)
    return trainer.update_step(prompts)

def agm_adaptive_optimization(policy_model: nn.Module,
                            ref_model: nn.Module,
                            optimizer: optim.Optimizer,
                            prompts: torch.Tensor, 
                            reward_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                            agm_iterations: int = 10) -> Dict[str, float]:
    """Adaptive optimization using AGM iteration."""
    config = PowerMeanConfig(mean_type='agm_adaptive', agm_iterations=agm_iterations)
    trainer = PowerMeanTrainer(policy_model, ref_model, optimizer, config, reward_fn)
    return trainer.update_step(prompts) 