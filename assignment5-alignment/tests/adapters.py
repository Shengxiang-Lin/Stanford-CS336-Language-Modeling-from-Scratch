from __future__ import annotations

import json
import os
import random
import re
from typing import Any, Callable, Literal

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase


def run_tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Tensor]:
    """Tokenize the prompt and output strings, and construct a mask that is 1
    for the response tokens and 0 for other tokens (prompt or padding).

    Args:
        prompt_strs: list[str], the prompt strings.
        output_strs: list[str], the output strings.
        tokenizer: PreTrainedTokenizer, the tokenizer to use.

    Returns:
        dict[str, torch.Tensor]:
            "input_ids": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                the tokenized prompt and output strings, with the final token sliced off.
            "labels": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                shifted input_ids (i.e., the input_ids without the first token).
            "response_mask": torch.Tensor of shape (batch_size, max(prompt_and_output_lens) - 1):
                a mask on the response tokens in `labels`.
    """
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt_and_output = [
        prompt_str + output_str
        for prompt_str, output_str in zip(prompt_strs, output_strs, strict=True)
    ]
    tokenized = tokenizer(
        prompt_and_output,
        padding=True,
        return_tensors="pt",
        add_special_tokens=True,
    )
    prompt_tokenized = tokenizer(
        prompt_strs,
        add_special_tokens=False,
    )["input_ids"]

    full_input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"].bool()
    input_ids = full_input_ids[:, :-1]
    labels = full_input_ids[:, 1:]
    label_attention_mask = attention_mask[:, 1:]
    response_mask = torch.zeros_like(labels, dtype=torch.bool)

    bos_token_id = tokenizer.bos_token_id
    for row_idx, prompt_ids in enumerate(prompt_tokenized):
        prompt_len = len(prompt_ids)
        if bos_token_id is not None and full_input_ids[row_idx, 0].item() == bos_token_id:
            prompt_len += 1

        seq_len = int(attention_mask[row_idx].sum().item())
        response_start = max(prompt_len - 1, 0)
        response_end = max(seq_len - 1, 0)
        response_mask[row_idx, response_start:response_end] = True

    response_mask &= label_attention_mask
    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
    }


def run_compute_group_normalized_rewards(
    reward_fn: Callable,
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute rewards for each group of rollout responses, 
    normalized by the group size.

    For more on GRPO, see:
        DeepSeekMath: https://arxiv.org/abs/2402.03300
        DeepSeek-R1: https://arxiv.org/abs/2501.12948

    Args:
        reward_fn: Callable[[str, str], dict[str, float]], 
            scores the rollout responses against the ground truths, 
            producing a dict with keys 
            "reward", "format_reward", and "answer_reward".
        rollout_responses: list[str], rollouts from the policy. 
            The length of this list is 
            `rollout_batch_size = n_prompts_per_rollout_batch * group_size`.
        repeated_ground_truths: list[str], the ground truths for the examples. 
            The length of this list is `rollout_batch_size`, 
            because the ground truth for each example is repeated `group_size` times.
        group_size: int, number of rollouts per group.
        advantage_eps: float, epsilon to avoid division by zero
            during group normalization.
        normalize_by_std: bool, whether to normalize the rewards by
            std(rewards).

    Returns:
        tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
            torch.Tensor of shape (rollout_batch_size,): 
                group-normalized rewards for each rollout response.
            torch.Tensor of shape (rollout_batch_size,): 
                raw rewards for each rollout response.
            dict[str, float]: metadata for the rewards of the rollout batch.
                You may choose what you wish to log here
                (some statistics of the rewards, etc.).
    """
    reward_outputs = [
        reward_fn(response, ground_truth)
        for response, ground_truth in zip(
            rollout_responses, repeated_ground_truths, strict=True
        )
    ]
    raw_rewards = torch.tensor(
        [reward_output["reward"] for reward_output in reward_outputs],
        dtype=torch.float32,
    )

    grouped_rewards = raw_rewards.view(-1, group_size)
    group_means = grouped_rewards.mean(dim=1, keepdim=True)

    if normalize_by_std:
        group_stds = grouped_rewards.std(dim=1, keepdim=True, unbiased=True)
        normalized_grouped_rewards = (grouped_rewards - group_means) / (
            group_stds + advantage_eps
        )
    else:
        normalized_grouped_rewards = grouped_rewards - group_means

    metadata = {
        "reward_mean": float(raw_rewards.mean().item()),
        "reward_std": float(raw_rewards.std(unbiased=False).item()),
        "format_reward_mean": float(
            sum(reward_output["format_reward"] for reward_output in reward_outputs)
            / len(reward_outputs)
        ),
        "answer_reward_mean": float(
            sum(reward_output["answer_reward"] for reward_output in reward_outputs)
            / len(reward_outputs)
        ),
    }
    return normalized_grouped_rewards.reshape(-1), raw_rewards, metadata


def run_compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Get the entropy of the logits (i.e., entropy of the final dimension)."""
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.softmax(logits, dim=-1)
    return -(probs * log_probs).sum(dim=-1)


def run_get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
) -> torch.Tensor:
    """Get the conditional log-probs of the response given the prompt,
        and optionally the entropy of the next token predictions.

    Args:
        model: PreTrainedModel, the model to score.
        input_ids: torch.Tensor of shape (batch_size, sequence_length):
            the tokenized prompt and output.
        labels: torch.Tensor of shape (batch_size, sequence_length):
            shifted input_ids.
        return_token_entropy: bool, whether to return the entropy of the
            next token predictions.

    Returns:
        dict[str, torch.Tensor]:
            "log_probs": torch.Tensor of shape (batch_size, sequence_length):
                the conditional log-probs of the response given the prompt.
                Note that we have not masked out the token indices corresponding
                to the prompt or padding; that is done in the train loop.
            "token_entropy": Optional[torch.Tensor] of shape (batch_size, sequence_length):
                the entropy of the next token predictions. As with the log-probs,
                we have not masked out the token indices corresponding to the prompt
                or padding; that is done in the train loop.
    """
    logits = model(input_ids=input_ids).logits
    log_probs = torch.log_softmax(logits, dim=-1)
    gathered_log_probs = torch.gather(
        log_probs, dim=-1, index=labels.unsqueeze(-1)
    ).squeeze(-1)
    output = {"log_probs": gathered_log_probs}
    if return_token_entropy:
        output["token_entropy"] = run_compute_entropy(logits)
    return output


def run_compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Compute policy gradient loss using either raw rewards or advantages.

    Args:
        raw_rewards_or_advantages: torch.Tensor of shape (batch_size, 1): 
            the raw rewards or advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.

    Returns:
        torch.Tensor of shape (batch_size, sequence_length): 
            the policy gradient per-token loss.
    """
    return -raw_rewards_or_advantages * policy_log_probs


def run_compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the GRPO-Clip loss.

    Args:
        advantages: torch.Tensor of shape (batch_size, 1): 
            the advantages for each rollout response.
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.
        old_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the old policy.
        cliprange: float, the clip range for the ratio.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]:
            torch.Tensor of shape (batch_size, sequence_length): 
                the GRPO-Clip per-token loss.
            dict[str, torch.Tensor]: metadata for the GRPO-Clip loss 
                (used to compute clip fraction).
    """
    ratio = torch.exp(policy_log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
    unclipped_objective = ratio * advantages
    clipped_objective = clipped_ratio * advantages
    clipped_loss = -torch.minimum(unclipped_objective, clipped_objective)
    metadata = {
        "ratio": ratio,
        "clipped_ratio": clipped_ratio,
        "is_clipped": ratio.ne(clipped_ratio),
    }
    return clipped_loss, metadata


def run_compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: str,
    raw_rewards: torch.Tensor,
    advantages: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Wrapper that delegates to the appropriate policy gradient loss function above.
    """
    if loss_type == "no_baseline":
        return run_compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs), {}
    if loss_type == "reinforce_with_baseline":
        return run_compute_naive_policy_gradient_loss(advantages, policy_log_probs), {}
    if loss_type == "grpo_clip":
        return run_compute_grpo_clip_loss(
            advantages=advantages,
            policy_log_probs=policy_log_probs,
            old_log_probs=old_log_probs,
            cliprange=cliprange,
        )
    raise ValueError(f"Unknown loss_type: {loss_type}")


def run_masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    """Compute the mean of the tensor along a dimension,
    considering only the elements with mask value 1.

    Args:
        tensor: torch.Tensor, the tensor to compute the mean of.
        mask: torch.Tensor, the mask. We only take the mean over
            the elements with mask value 1.
        dim: int | None, the dimension to compute the mean along.
            If None, sum over all non-masked elements and average
            by their total count.

    Returns:
        torch.Tensor, the mean of the tensor along the specified
            dimension, considering only the elements with mask value 1.
    """
    mask = mask.to(dtype=tensor.dtype)
    if dim is None:
        numerator = (tensor * mask).sum()
        denominator = mask.sum()
    else:
        numerator = (tensor * mask).sum(dim=dim)
        denominator = mask.sum(dim=dim)
    return numerator / denominator

def run_sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: int | None = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch.
    """
    per_token_loss = -policy_log_probs
    per_example_loss = run_masked_normalize(
        tensor=per_token_loss,
        mask=response_mask,
        dim=-1,
        normalize_constant=normalize_constant,
    )
    loss = per_example_loss.mean() / gradient_accumulation_steps
    loss.backward()
    metadata = {
        "num_response_tokens": response_mask.sum(),
    }
    return loss.detach(), metadata

    
def run_grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the policy gradient loss and backprop its gradients for a microbatch.

    Args:
        policy_log_probs: torch.Tensor of shape (batch_size, sequence_length): 
            the log-probs of the policy.
        response_mask: torch.Tensor of shape (batch_size, sequence_length): 
            the mask for the response.
        gradient_accumulation_steps: int, the number of gradient accumulation steps.
        loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"], 
            the type of loss function to use.
        raw_rewards: torch.Tensor | None, the raw rewards for each rollout response.
            Needed for loss_type="no_baseline".
        advantages: torch.Tensor | None, the advantages for each rollout response.
            Needed for loss_type in {"reinforce_with_baseline", "grpo_clip"}.
        old_log_probs: torch.Tensor | None, the log-probs of the old policy.
            Needed for loss_type="grpo_clip".
        cliprange: float | None, the clip range for the ratio. 
            Needed for loss_type="grpo_clip".
        constant_normalize_factor: int | None, provided if we want to sum over 
            the sequence dimension and normalize by this constant factor
            (as in Dr. GRPO).

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]]: 
            the policy gradient loss and its metadata.
    """
    per_token_loss, metadata = run_compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )
    per_example_loss = run_masked_mean(per_token_loss, response_mask, dim=-1)
    loss = per_example_loss.mean() / gradient_accumulation_steps
    loss.backward()
    return loss.detach(), metadata


def run_masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    normalize_constant: float = 1.0,
) -> torch.Tensor:
    """Sum over a dimension and normalize by a constant,
    considering only the elements with mask value 1.

    Args:
        tensor: torch.Tensor, the tensor to sum and normalize.
        mask: torch.Tensor, the mask. We only consider elements
            with mask value 1.
        dim: int | None, the dimension to sum along before
            normalization. If None, sum over all dimensions.
        normalize_constant: float, the constant to divide by
            for normalization.

    Returns:
        torch.Tensor, the normalized sum, where masked elements
            (mask=0) don't contribute to the sum.
    """
    mask = mask.to(dtype=tensor.dtype)
    if dim is None:
        return (tensor * mask).sum() / normalize_constant
    return (tensor * mask).sum(dim=dim) / normalize_constant


"""
The below adapters are used in the optional 
RLHF / safety part of the Alignment assignment.
"""


def get_packed_sft_dataset(
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: str | os.PathLike,
    seq_length: int,
    shuffle: bool,
) -> Dataset:
    """
    Given a tokenizer and a path to a dataset with instruction-tuning examples,
    construct a PyTorch Dataset for language modeling. The examples should be
    packed, i.e., all sequences in the dataset are of a constant length (`seq_length`).

    Args:
        tokenizer: transformers.PreTrainedTokenizerBase
            Transformers tokenizer to use in tokenizing and encoding text.
        dataset_path: str
            Path to file with instruction-tuning examples.
        seq_length: int
            Number of tokens to include in each example.
        shuffle: bool
            If true, shuffle the documents before packing them into examples.

    Returns:
        PyTorch Dataset for language modeling. Each example in this dataset is a dictionary of
        with keys "input_ids" and "labels" (both tensors of shape (seq_length, )).
        "input_ids" contains the token IDs for the language modeling inputs, and "labels" contains
        the token IDs for the language modeling labels.
    """
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    with open(dataset_path) as f:
        records = [json.loads(line) for line in f]

    if shuffle:
        random.shuffle(records)

    all_token_ids: list[int] = []
    for record in records:
        formatted_example = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n"
            f"{record['prompt']}\n\n"
            "### Response:\n"
            f"{record['response']}"
        )
        token_ids = tokenizer.encode(formatted_example, add_special_tokens=True)
        if tokenizer.eos_token_id is not None:
            token_ids = token_ids + [tokenizer.eos_token_id]
        all_token_ids.extend(token_ids)

    examples = []
    for start_idx in range(0, len(all_token_ids) - 1, seq_length):
        end_idx = start_idx + seq_length + 1
        if end_idx > len(all_token_ids):
            break
        chunk = all_token_ids[start_idx:end_idx]
        examples.append(
            {
                "input_ids": torch.tensor(chunk[:-1], dtype=torch.long),
                "labels": torch.tensor(chunk[1:], dtype=torch.long),
            }
        )

    class PackedSFTDataset(Dataset):
        def __init__(self, packed_examples: list[dict[str, Tensor]]) -> None:
            self._packed_examples = packed_examples

        def __len__(self) -> int:
            return len(self._packed_examples)

        def __getitem__(self, idx: int) -> dict[str, Tensor]:
            return self._packed_examples[idx]

    return PackedSFTDataset(examples)


def run_iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
):
    """
    Given a PyTorch Dataset, return an iterable over batches of size `batch_size`.
    Iterating through the returned iterable should constitute one epoch over the Dataset.

    Args:
        dataset: Dataset
            Dataset to emit batches from.
        batch_size: int
            Number of examples to include per batch.
        shuffle: bool
            If true, shuffle examples before batching them.

    Returns:
        Iterable over batches, where each batch has size `batch_size`.
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def run_parse_mmlu_response(
    mmlu_example: dict[str, Any],
    model_output: str,
) -> str | None:
    """
    Given an MMLU example and a model output, parse the model output into a
    predicted option letter (i.e., 'A', 'B', 'C', or 'D'). If the model output
    cannot be parsed into a prediction option letter, return None.

    mmlu_example: dict[str, Any]
        Dictionary with an MMLU example. Contains the following keys:
        - "subject": str with the subject of the question.
        - "question": str with the text of the question.
        - "options": list[str] with the four answer options (in order).
                     The first option refers to letter "A", the second to "B", etc.
        - "answer": str with the option of the correct answer (e.g., "A")
    model_output: str
        str with the model's output to the MMLU example.

    Returns:
        str (one of "A", "B", "C", or "D") if the model output can be parsed into a prediction,
        else None.
    """
    match = re.search(r"\b([ABCD])\b", model_output.upper())
    return match.group(1) if match else None


def run_parse_gsm8k_response(
    model_output: str,
) -> str | None:
    """
    Given a GSM8K model output, parse the model output into a predicted numeric answer by
    taking the last number that occurs in the output.

    model_output: str
        str with the model's output to a GSM8K example.

    Returns:
        str with the predicted numeric answer if the model output can be parsed into a prediction,
        else None.
    """
    matches = re.findall(r"-?\d[\d,]*(?:\.\d+)?", model_output)
    if not matches:
        return None
    return matches[-1].replace(",", "")


def run_compute_per_instance_dpo_loss(
    lm: torch.nn.Module,
    lm_ref: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    """
    Given two language models (`lm`, and the "reference model" `lm_ref`),
    their tokenizer, the DPO beta hyperparameter, a prompt and a pair
    of responses to the prompt, computes the value of the DPO loss for this example.

    lm: torch.nn.Module
        Language model being trained.
    lm_ref: torch.nn.Module
        Reference language model.
    tokenizer: PreTrainedTokenizerBase
        Tokenizer for both language models.
    beta: float
        DPO beta hyperparameter.
    prompt: str
        Prompt for this instance of preference pair.
    response_chosen: str
        Preferred response to the prompt.
    response_rejected: str
        Rejected response to the prompt.

    Returns:
        torch.Tensor with the DPO loss for this example.
    """
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    def format_alpaca_example(response: str) -> str:
        return (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n"
            f"{prompt}\n\n"
            "### Response:\n"
            f"{response}"
        )

    def tokenize_dpo_example(response: str) -> dict[str, Tensor]:
        formatted_example = format_alpaca_example(response)
        if tokenizer.eos_token is not None:
            formatted_example += tokenizer.eos_token

        full_input_ids = tokenizer(
            formatted_example,
            return_tensors="pt",
            add_special_tokens=True,
        )["input_ids"]
        prompt_only_input_ids = tokenizer(
            format_alpaca_example(""),
            return_tensors="pt",
            add_special_tokens=True,
        )["input_ids"]

        input_ids = full_input_ids[:, :-1]
        labels = full_input_ids[:, 1:]
        response_mask = torch.zeros_like(labels, dtype=torch.bool)
        response_mask[:, prompt_only_input_ids.shape[1] - 1 : full_input_ids.shape[1] - 1] = True
        return {
            "input_ids": input_ids,
            "labels": labels,
            "response_mask": response_mask,
        }

    chosen_batch = tokenize_dpo_example(response_chosen)
    rejected_batch = tokenize_dpo_example(response_rejected)

    chosen_log_probs = run_get_response_log_probs(
        model=lm,
        input_ids=chosen_batch["input_ids"],
        labels=chosen_batch["labels"],
        return_token_entropy=False,
    )["log_probs"]
    chosen_ref_log_probs = run_get_response_log_probs(
        model=lm_ref,
        input_ids=chosen_batch["input_ids"],
        labels=chosen_batch["labels"],
        return_token_entropy=False,
    )["log_probs"]
    rejected_log_probs = run_get_response_log_probs(
        model=lm,
        input_ids=rejected_batch["input_ids"],
        labels=rejected_batch["labels"],
        return_token_entropy=False,
    )["log_probs"]
    rejected_ref_log_probs = run_get_response_log_probs(
        model=lm_ref,
        input_ids=rejected_batch["input_ids"],
        labels=rejected_batch["labels"],
        return_token_entropy=False,
    )["log_probs"]

    chosen_score = (
        chosen_log_probs * chosen_batch["response_mask"].to(chosen_log_probs.dtype)
    ).sum(dim=1)
    chosen_ref_score = (
        chosen_ref_log_probs
        * chosen_batch["response_mask"].to(chosen_ref_log_probs.dtype)
    ).sum(dim=1)
    rejected_score = (
        rejected_log_probs * rejected_batch["response_mask"].to(rejected_log_probs.dtype)
    ).sum(dim=1)
    rejected_ref_score = (
        rejected_ref_log_probs
        * rejected_batch["response_mask"].to(rejected_ref_log_probs.dtype)
    ).sum(dim=1)

    preference_logit = beta * (
        (chosen_score - chosen_ref_score) - (rejected_score - rejected_ref_score)
    )
    return -torch.nn.functional.logsigmoid(preference_logit).mean()
