from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class EdgeStats:
    n: int = 0
    w: float = 0.0  # total value (root-perspective for AZBSMCTS)
    p: float = 0.0  # prior probability (for PUCT)

    @property
    def q(self) -> float:
        return 0.0 if self.n == 0 else self.w / self.n


@dataclass
class Node:
    obs_key: str
    player_to_act: int
    is_expanded: bool = False
    edges: Dict[int, EdgeStats] = field(default_factory=dict)
    legal_actions: List[int] = field(default_factory=list)
    n: int = 0  # node visit count


class BeliefTree:
    """
    Belief tree keyed by the agent's observation string (player-relative, partial info).
    """

    def __init__(self):
        self.nodes: Dict[str, Node] = {}

    def get_or_create(self, obs_key: str, player_to_act: int) -> Node:
        node = self.nodes.get(obs_key)
        if node is None:
            node = Node(obs_key=obs_key, player_to_act=player_to_act)
            self.nodes[obs_key] = node
        return node
