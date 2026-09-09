from pathlib import Path

import torch


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint must contain a state dictionary")

    state_dict = {}
    for key, value in checkpoint.items():
        while key.startswith("module."):
            key = key[len("module."):]
        if key.startswith("model."):
            key = key[len("model."):]
        # Ignore parameters from the unused auxiliary branch in older
        # checkpoints so those weights remain loadable.
        if key.startswith("tdem."):
            continue
        # The original BasicIRSTD wrapper stored this loss-only buffer beside
        # the model parameters. It is not part of standalone BCR-Net.
        if key == "lee_bce.pos_weight":
            continue
        state_dict[key] = value
    return state_dict


def load_checkpoint(model, path, map_location="cpu", strict=True):
    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=False)
    state_dict = _extract_state_dict(checkpoint)
    incompatible = model.load_state_dict(state_dict, strict=strict)
    return checkpoint, incompatible
