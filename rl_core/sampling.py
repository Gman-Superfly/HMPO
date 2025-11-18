from typing import Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical


def compute_log_probs_batch(
    model: nn.Module,
    full_sequences: torch.Tensor,
    prompt_length: int,
    response_lengths: torch.Tensor,
) -> torch.Tensor:
    """Compute log probabilities summed over response tokens for a batch.

    Args:
        model: Autoregressive language model producing logits.
        full_sequences: Concatenated prompts+responses (batch_size, total_len).
        prompt_length: Length of the prompt segment.
        response_lengths: Length of each response segment (batch_size,).

    Returns:
        Tensor of shape (batch_size,) with summed log-probs over response tokens.
    """
    assert full_sequences.size(0) == response_lengths.size(0), "Batch size mismatch"
    assert prompt_length > 0, "Invalid prompt length"

    device = full_sequences.device

    # Autoregressive shift
    input_ids = full_sequences[:, :-1]  # (batch_size, seq_len-1)
    target_ids = full_sequences[:, 1:]  # (batch_size, seq_len-1)

    # Forward pass
    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
        logits = model(input_ids)  # (batch_size, seq_len-1, vocab_size)

    # Log probabilities
    log_probs = torch.log_softmax(logits, dim=-1)

    # Gather log-probs for actual targets
    target_expanded = target_ids.unsqueeze(-1)  # (batch_size, seq_len-1, 1)
    gathered_log_probs = log_probs.gather(-1, target_expanded).squeeze(-1)  # (batch_size, seq_len-1)

    # Mask only response positions [prompt_length, prompt_length + response_length)
    seq_positions = torch.arange(gathered_log_probs.size(1), device=device).unsqueeze(0)
    response_start = prompt_length
    response_end = response_start + response_lengths.unsqueeze(1)
    response_mask = (seq_positions >= response_start) & (seq_positions < response_end)

    masked_log_probs = gathered_log_probs * response_mask.float()
    summed_log_probs = masked_log_probs.sum(dim=1)  # (batch_size,)

    assert not torch.any(torch.isnan(summed_log_probs)), "NaN in log probabilities"
    assert not torch.any(torch.isinf(summed_log_probs)), "Inf in log probabilities"
    return summed_log_probs


def sample_responses_batch(
    ref_model: nn.Module,
    prompts: torch.Tensor,
    group_size: int,
    max_length: int,
    eos_token: int,
    pad_token: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized sampling of responses from a reference model.

    Args:
        ref_model: Reference (frozen) model used for sampling.
        prompts: Batch of prompts (batch_size, prompt_len).
        group_size: Number of responses per prompt.
        max_length: Maximum response length.
        eos_token: End-of-sequence token id.
        pad_token: Padding token id.

    Returns:
        responses: (batch_size*group_size, <=max_length)
        lengths: (batch_size*group_size,)
        full_sequences: concatenated prompts+responses
    """
    assert prompts.size(0) > 0, "Empty prompt batch"

    device = prompts.device
    expanded_prompts = prompts.repeat_interleave(group_size, dim=0)
    total_sequences = expanded_prompts.size(0)

    current = expanded_prompts.clone()
    responses = torch.full((total_sequences, max_length), pad_token, dtype=torch.long, device=device)
    active = torch.ones(total_sequences, dtype=torch.bool, device=device)
    lengths = torch.zeros(total_sequences, dtype=torch.long, device=device)

    ref_model.eval()
    with torch.no_grad():
        for step in range(max_length):
            logits = ref_model(current)[:, -1, :]  # (total_sequences, vocab_size)
            dist = Categorical(logits=logits)
            next_tokens = dist.sample()  # (total_sequences,)
            next_tokens[~active] = pad_token

            responses[:, step] = next_tokens
            current = torch.cat([current, next_tokens.unsqueeze(1)], dim=1)

            just_finished = (next_tokens == eos_token) & active
            lengths[just_finished] = step + 1
            active = active & (next_tokens != eos_token)

            if not active.any():
                break

        # Any sequences that never emitted EOS
        lengths[active] = max_length

    actual_max_len = lengths.max().item()
    responses = responses[:, :actual_max_len]
    full_sequences = torch.cat([expanded_prompts, responses], dim=1)

    assert torch.all(lengths > 0), "Zero-length responses detected"
    return responses, lengths, full_sequences


