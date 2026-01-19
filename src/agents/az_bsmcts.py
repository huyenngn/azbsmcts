from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import pyspiel
import torch

from agents.base import BaseAgent, PolicyTargetMixin
from belief.samplers.base import DeterminizationSampler
from belief.tree import BeliefTree, EdgeStats, Node
from nets.tiny_policy_value import TinyPolicyValueNet, get_shared_az_model
from utils.softmax import softmax_np


class AZBSMCTSAgent(BaseAgent, PolicyTargetMixin):
    """
    AZ-guided BS-MCTS.

    Strict rules enforced:
    - No default clone determinization (must pass a belief-based sampler).
    - No observation_tensor()/observation_string() without player id.
    - NN input is from SIDE-TO-MOVE perspective: observation_tensor(state.current_player()).
    - NN value is assumed to be SIDE-TO-MOVE perspective.
      We convert to ROOT-player perspective for backup.
    """

    def __init__(
        self,
        player_id: int,
        num_actions: int,
        obs_size: int,
        sampler: DeterminizationSampler,
        c_puct: float = 1.5,
        T: int = 64,
        S: int = 8,
        seed: int = 0,
        device: str = "cpu",
        net: Optional[TinyPolicyValueNet] = None,
        model_path: Optional[str] = None,
    ):
        super().__init__(
            player_id=player_id, num_actions=num_actions, seed=seed
        )
        self.tree = BeliefTree()
        self.sampler = sampler
        self.c_puct = float(c_puct)
        self.T = int(T)
        self.S = int(S)

        self.device = device
        if net is None:
            self.net = get_shared_az_model(
                obs_size=obs_size,
                num_actions=num_actions,
                model_path=model_path,
                device=device,
            )
        else:
            self.net = net.to(device)
            self.net.eval()

        self._obs_size = int(obs_size)

    def _state_tensor_side_to_move(self, state: pyspiel.State) -> torch.Tensor:
        side = state.current_player()

        # Strict: always use explicit player id
        obs = self.obs_tensor(state, side)
        if obs.size != self._obs_size:
            raise ValueError(
                f"obs_size mismatch: expected {self._obs_size}, got {obs.size}"
            )
        return torch.from_numpy(obs).to(self.device)

    def _expand(self, node: Node, state: pyspiel.State):
        node.is_expanded = True
        node.legal_actions = list(state.legal_actions())
        for a in node.legal_actions:
            node.edges.setdefault(a, EdgeStats())

        with torch.no_grad():
            x = self._state_tensor_side_to_move(state).unsqueeze(0)
            logits, _v = self.net(x)
            logits = logits.squeeze(0).detach().cpu().numpy()

        mask = np.full((self.num_actions,), -1e9, dtype=np.float32)
        for a in node.legal_actions:
            mask[a] = 0.0
        priors = softmax_np(logits + mask)

        for a in node.legal_actions:
            node.edges[a].p = float(priors[a])

    def _leaf_value_root_perspective(self, state: pyspiel.State) -> float:
        with torch.no_grad():
            x = self._state_tensor_side_to_move(state).unsqueeze(0)
            _logits, v = self.net(x)
            v_cur = float(v.item())

        # Convert side-to-move -> root perspective
        return v_cur if state.current_player() == self.player_id else -v_cur

    def _puct(self, parent: Node, edge: EdgeStats) -> float:
        q = edge.q
        u = self.c_puct * edge.p * math.sqrt(parent.n + 1.0) / (1.0 + edge.n)
        return q + u

    def _search(self, state: pyspiel.State) -> float:
        if state.is_terminal():
            return float(state.returns()[self.player_id])

        node = self.tree.get_or_create(
            self.obs_key(state, self.player_id), state.current_player()
        )
        node.n += 1

        if not node.is_expanded:
            self._expand(node, state)
            return self._leaf_value_root_perspective(state)

        legal_now = set(state.legal_actions())
        best_a = None
        best_s = -1e18
        for a, e in node.edges.items():
            if a not in legal_now:
                continue
            s = self._puct(node, e)
            if s > best_s:
                best_s = s
                best_a = a

        if best_a is None:
            return self._leaf_value_root_perspective(state)

        state.apply_action(best_a)
        v_root = self._search(state)

        edge = node.edges[best_a]
        edge.n += 1
        edge.w += v_root
        return v_root

    def _root_visit_policy(self, root: Node, temperature: float) -> np.ndarray:
        pi = np.zeros((self.num_actions,), dtype=np.float32)
        if not root.edges:
            return pi

        actions = list(root.edges.keys())
        visits = np.array([root.edges[a].n for a in actions], dtype=np.float32)

        if temperature <= 1e-8:
            pi[actions[int(np.argmax(visits))]] = 1.0
            return pi

        vt = np.power(visits + 1e-8, 1.0 / float(temperature))
        probs = vt / float(np.sum(vt))
        for a, p in zip(actions, probs):
            pi[a] = float(p)
        return pi

    def select_action(self, state: pyspiel.State) -> int:
        a, _pi = self.select_action_with_pi(state, temperature=1e-8)
        return a

    def select_action_with_pi(
        self, state: pyspiel.State, temperature: float = 1.0
    ) -> Tuple[int, np.ndarray]:
        root = self.tree.get_or_create(
            self.obs_key(state, self.player_id), state.current_player()
        )
        if not root.is_expanded:
            self._expand(root, state)

        for _ in range(self.T):
            gamma = self.sampler.sample(state, self.rng)
            for _ in range(self.S):
                self._search(gamma.clone())

        pi = self._root_visit_policy(root, temperature=float(temperature))

        legal = set(state.legal_actions())
        probs = pi.copy()
        for a in range(self.num_actions):
            if a not in legal:
                probs[a] = 0.0

        s = float(np.sum(probs))
        if s <= 0:
            best_a = max(
                (a for a in legal),
                key=lambda a: root.edges.get(a, EdgeStats()).n,
            )
            return int(best_a), pi

        probs /= s
        action = int(np.random.choice(np.arange(self.num_actions), p=probs))
        return action, pi
