from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyPolicyValueNet(nn.Module):
    """
    Minimal policy/value network.
    Input: flattened observation tensor
    Output: policy logits over action space, and scalar value in [-1,1]
    """

    def __init__(self, obs_size: int, num_actions: int, hidden: int = 256):
        super().__init__()
        self.obs_size = obs_size
        self.num_actions = num_actions

        self.fc1 = nn.Linear(obs_size, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.policy = nn.Linear(hidden, num_actions)
        self.value = nn.Linear(hidden, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.policy(x)
        v = torch.tanh(self.value(x))
        return logits, v


class _AZShared(TypedDict):
    net: Optional[TinyPolicyValueNet]
    device: Optional[str]
    path: Optional[str]
    obs_size: Optional[int]
    num_actions: Optional[int]


_AZ_SHARED: _AZShared = {
    "net": None,
    "device": None,
    "path": None,
    "obs_size": None,
    "num_actions": None,
}


def get_shared_az_model(
    obs_size: int,
    num_actions: int,
    model_path: Optional[str] = None,
    device: Optional[str] = None,
) -> TinyPolicyValueNet:
    """
    Singleton net cache: keeps eval model around for agents/eval/training loops.
    """
    if device is None:
        device = os.environ.get("AZ_DEVICE", "cpu")
    if model_path is None:
        model_path = os.environ.get("AZ_MODEL_PATH", "models/model.pt")

    if (
        _AZ_SHARED["net"] is not None
        and _AZ_SHARED["device"] == device
        and _AZ_SHARED["path"] == model_path
        and _AZ_SHARED["obs_size"] == obs_size
        and _AZ_SHARED["num_actions"] == num_actions
    ):
        return _AZ_SHARED["net"]  # type: ignore[return-value]

    net = TinyPolicyValueNet(obs_size=obs_size, num_actions=num_actions).to(
        device
    )
    net.eval()

    p = Path(model_path)
    if p.exists():
        state_dict = torch.load(str(p), map_location=device)
        net.load_state_dict(state_dict)

    _AZ_SHARED["net"] = net
    _AZ_SHARED["device"] = device
    _AZ_SHARED["path"] = model_path
    _AZ_SHARED["obs_size"] = obs_size
    _AZ_SHARED["num_actions"] = num_actions
    return net
