"""Small inference utilities shared by the public entry points."""

import torch


def count_parameters(model: torch.nn.Module) -> int:
    """Return the number of trainable parameters in a model."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def setup_rng(
    seeds: list[int],
    device: torch.device | str,
) -> list[torch.Generator]:
    """Create one deterministic PyTorch generator for each seed."""
    return [torch.Generator(device=device).manual_seed(seed) for seed in seeds]
