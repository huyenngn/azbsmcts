from __future__ import annotations

import math

import pyspiel

from agents.base import BaseAgent
from belief.samplers.base import DeterminizationSampler
from belief.tree import BeliefTree, EdgeStats, Node


class BSMCTSAgent(BaseAgent):
    """
    Baseline: dumb BS-MCTS-ish agent.
    IMPORTANT:
    - Requires an explicit DeterminizationSampler (no cloning real state).
    """

    def __init__(
        self,
        player_id: int,
        num_actions: int,
        sampler: DeterminizationSampler,
        c_uct: float = 1.4,
        T: int = 64,
        S: int = 8,
        rollout_limit: int = 512,
        seed: int = 0,
    ):
        super().__init__(
            player_id=player_id, num_actions=num_actions, seed=seed
        )
        self.tree = BeliefTree()
        self.sampler = sampler
        self.c_uct = float(c_uct)
        self.T = int(T)
        self.S = int(S)
        self.rollout_limit = int(rollout_limit)

    def select_action(self, state: pyspiel.State) -> int:
        root = self.tree.get_or_create(
            self.obs_key(state, self.player_id), state.current_player()
        )

        if not root.is_expanded:
            self._expand(root, state)

        for _ in range(self.T):
            gamma = self.sampler.sample(state, self.rng)
            for _ in range(self.S):
                self._search(gamma.clone())

        return max(root.edges.items(), key=lambda kv: kv[1].n)[0]

    def _expand(self, node: Node, state: pyspiel.State):
        node.is_expanded = True
        node.legal_actions = list(state.legal_actions())
        for a in node.legal_actions:
            node.edges.setdefault(a, EdgeStats())

    def _uct(self, parent: Node, edge: EdgeStats) -> float:
        q = edge.q
        u = self.c_uct * math.sqrt(math.log(parent.n + 1.0) / (edge.n + 1.0))
        return q + u

    def _rollout(self, state: pyspiel.State) -> float:
        steps = 0
        while (not state.is_terminal()) and steps < self.rollout_limit:
            la = state.legal_actions()
            if not la:
                break
            state.apply_action(self.rng.choice(la))
            steps += 1

        if state.is_terminal():
            return float(state.returns()[self.player_id])
        return 0.0

    def _search(self, state: pyspiel.State) -> float:
        if state.is_terminal():
            return float(state.returns()[self.player_id])

        node = self.tree.get_or_create(
            self.obs_key(state, self.player_id), state.current_player()
        )
        node.n += 1

        if not node.is_expanded:
            self._expand(node, state)
            return self._rollout(state.clone())

        legal_now = set(state.legal_actions())
        best_a = None
        best_s = -1e18
        for a, e in node.edges.items():
            if a not in legal_now:
                continue
            s = self._uct(node, e)
            if s > best_s:
                best_s = s
                best_a = a

        if best_a is None:
            return self._rollout(state.clone())

        state.apply_action(best_a)
        v = self._search(state)

        edge = node.edges[best_a]
        edge.n += 1
        edge.w += v
        return v
