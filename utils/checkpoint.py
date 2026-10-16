"""Checkpoint loading helpers."""
import torch
import torch.nn as nn

def load_model_checkpoint(model: nn.Module, ckpt_path: str, map_location="cpu"):
    """
    Load a checkpoint and migrate legacy attribute names used before the paper-aligned cleanup.
    Legacy prefixes:
      gcn.*                  -> dtc_gcn.*
      gcn.cnn_branch.*       -> dtc_gcn.local_cnn_branch.*
      task_iq.*              -> gca.*
    The old unused scalar parameter `gamma` is intentionally discarded.
    """
    state = torch.load(ckpt_path, map_location=map_location)
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint format: {type(state).__name__}")

    migrated = {}
    for key, value in state.items():
        new_key = key
        if new_key.startswith("module."):
            new_key = new_key[len("module."):]
        if new_key == "gamma" or new_key.endswith(".gamma"):
            continue
        if new_key.startswith("gcn.cnn_branch."):
            new_key = "dtc_gcn.local_cnn_branch." + new_key[len("gcn.cnn_branch."):]
        elif new_key.startswith("gcn."):
            new_key = "dtc_gcn." + new_key[len("gcn."):]
        elif new_key.startswith("task_iq."):
            new_key = "gca." + new_key[len("task_iq."):]
        migrated[new_key] = value

    result = model.load_state_dict(migrated, strict=False)
    print(f"Checkpoint loaded: {ckpt_path}")
    if result.missing_keys:
        print("Missing checkpoint keys (first 20):", result.missing_keys[:20])
    if result.unexpected_keys:
        print("Unexpected checkpoint keys (first 20):", result.unexpected_keys[:20])
    return result
