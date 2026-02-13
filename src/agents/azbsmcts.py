"""AlphaZero-guided Belief-State MCTS (AZBSMCTS).

Structurally identical to BS-MCTS (Algorithm 1 from Wang et al., 2015) but
replaces random rollouts with neural-network evaluation:

* **Expansion** — the NN policy head provides per-(γ, action) priors ``p``.
* **Leaf evaluation** — the NN value head replaces the rollout.
* **Selection** — PUCT (Q + c·P·√N/(1+n)) instead of UCT.
* **Backpropagation** — sum-based ``w`` instead of running-average ``u``.
* **Action selection** — visit-count policy vector (for training targets).

Everything else — the T×S sampling loop, the per-γ belief tree, Opponent
Guessing / Predicting — is inherited from ``BSMCTSAgent``.
"""

from __future__ import annotations

import typing as t

import numpy as np
import torch

import agents
import nets
from utils import softmax

if t.TYPE_CHECKING:
  import openspiel
  from belief import samplers, tree


class AZBSMCTSAgent(agents.MCTSAgent, agents.PolicyTargetMixin):
  """AlphaZero-guided BS-MCTS agent.

  Inherits the full BS-MCTS loop from ``BSMCTSAgent`` and overrides:
  * ``_expand`` — uses NN policy priors instead of plain legal-action init.
  * ``_search`` — uses PUCT selection and NN value instead of UCT + rollout.
  * ``select_action`` — returns argmax-visit action (greedy).
  * ``select_action_with_pi`` — returns (action, policy vector) for training.

  Parameters
  ----------
  game, player_id, sampler, T, S, seed, lambda_guess, lambda_predict,
  length_discount:
      Passed through to ``BSMCTSAgent``.
  obs_size : int
      Size of the observation tensor expected by the network.
  c_puct : float
      PUCT exploration constant (replaces ``c_uct``).
  device : str
      Torch device for NN inference.
  net : TinyPolicyValueNet | None
      Pre-loaded network.  Exactly one of ``net`` / ``model_path`` required.
  model_path : str | None
      Path to saved model weights.
  dirichlet_alpha, dirichlet_weight : float
      Dirichlet noise parameters for root exploration during training.
  """

  def __init__(
    self,
    game: openspiel.Game,
    player_id: int,
    obs_size: int,
    sampler: samplers.DeterminizationSampler,
    c_puct: float = 1.5,
    T: int = 100,
    S: int = 8,
    seed: int = 0,
    device: str = "cpu",
    net: nets.TinyPolicyValueNet | None = None,
    model_path: str | None = None,
    dirichlet_alpha: float = 0.03,
    dirichlet_weight: float = 0.25,
    length_discount: float = 0.999,
    lambda_guess: float = 1.0,
    lambda_predict: float = 1.0,
  ):
    super().__init__(
      game=game,
      player_id=player_id,
      sampler=sampler,
      c_uct=c_puct,
      T=T,
      S=S,
      seed=seed,
      lambda_guess=lambda_guess,
      lambda_predict=lambda_predict,
      length_discount=length_discount,
    )
    self.c_puct = float(c_puct)
    self.dirichlet_alpha = float(dirichlet_alpha)
    self.dirichlet_weight = float(dirichlet_weight)

    self.device = device
    if net is None:
      if model_path is None:
        raise ValueError("Either 'net' or 'model_path' must be provided")
      self.net = nets.get_shared_az_model(
        obs_size=obs_size,
        num_actions=game.num_distinct_actions(),
        model_path=model_path,
        device=device,
      )
    else:
      self.net = net.to(device)
      self.net.eval()

    self._obs_size = int(obs_size)

    # Pending leaf evaluations for batched NN inference.
    # Each entry: (gamma_key, node, state, path-from-root)
    # path = list of (node, action) pairs walked before reaching the leaf.
    self._pending_leaves: list[
      tuple[
        str,
        tree.Node,
        openspiel.State,
        list[tuple[tree.Node, str, int]],
      ]
    ] = []

  # -- NN helpers -----------------------------------------------------------

  def _state_tensor_side_to_move(self, state: openspiel.State) -> torch.Tensor:
    """Get observation tensor from current player's perspective."""
    side = state.current_player()
    obs = self.obs_tensor(state, side)
    if obs.size != self._obs_size:
      raise ValueError(
        f"obs_size mismatch: expected {self._obs_size}, got {obs.size}"
      )
    return torch.from_numpy(obs).to(self.device)

  def _nn_priors(self, state: openspiel.State) -> np.ndarray:
    """Run NN policy head and return masked softmax priors."""
    with torch.no_grad():
      x = self._state_tensor_side_to_move(state).unsqueeze(0)
      logits, _v = self.net(x)
      logits = logits.squeeze(0).detach().cpu().numpy()
    mask = np.full((self.game.num_distinct_actions(),), -1e9, dtype=np.float32)
    for a in state.legal_actions():
      mask[a] = 0.0
    return softmax.softmax_np(logits + mask)

  def _nn_value(self, state: openspiel.State) -> float:
    """Run NN value head and return scalar from side-to-move perspective."""
    with torch.no_grad():
      x = self._state_tensor_side_to_move(state).unsqueeze(0)
      _, v = self.net(x)
    return float(v.item())

  def _value_root_perspective(
    self, state: openspiel.State, value: float
  ) -> float:
    """Convert side-to-move value to root player's perspective."""
    return value if state.current_player() == self.player_id else -value

  # -- Override _expand: NN priors per (γ, action) --------------------------

  def _expand(
    self,
    gamma_key: str,
    gamma_state: openspiel.State,
    node: tree.Node,
    *,
    add_dirichlet: bool = False,
  ) -> None:
    """Expand γ inside belief-state B with NN policy priors.

    Initialises per-(γ, action) stats and stores the NN prior ``p``
    on each ``GammaActionStats``.  If the node hasn't been expanded at
    all yet (first γ to visit), also sets ``legal_actions``.
    """
    node.is_expanded = True
    legal = list(gamma_state.legal_actions())
    for a in legal:
      if a not in node.legal_actions:
        node.legal_actions.append(a)

    # NN priors for this determinization
    priors = self._nn_priors(gamma_state)

    if add_dirichlet and self.dirichlet_alpha > 0:
      dir_seed = self.rng.getrandbits(32)
      dir_rng = np.random.default_rng(dir_seed)
      noise = dir_rng.dirichlet([self.dirichlet_alpha] * len(legal))
      eps = self.dirichlet_weight
      for i, a in enumerate(legal):
        priors[a] = (1 - eps) * priors[a] + eps * noise[i]

    gs = node.get_or_create_gamma(gamma_key)
    for a in legal:
      a_stat = gs.get_or_create_action(a)
      a_stat.p = float(priors[a])

  def _selection(
    self,
    gamma_state: openspiel.State,
    node: tree.Node,
  ) -> int | None:
    legal = list(gamma_state.legal_actions())
    if not legal:
      return None

    if node.player_to_act == self.player_id:
      return max(legal, key=lambda a: node.puct_value(a, self.c_puct))
    probs = node.opponent_action_probs(legal, self.lambda_predict)
    return self.rng.choices(legal, weights=probs, k=1)[0]

  def _search(
    self,
    gamma_key: str,
    gamma_state: openspiel.State,
    node: tree.Node,
    *,
    batch_mode: bool = False,
    _path: list[tuple[tree.Node, str, int]] | None = None,
  ) -> float | None:
    """Recursive search on γ through belief node B using PUCT + NN.

    Mirrors ``BSMCTSAgent._search`` but:
    * Uses PUCT instead of UCT for player-node selection.
    * Uses NN value instead of rollout for leaf evaluation.
    * Backpropagates sum-based ``w`` instead of running-average ``u``.
    * Supports batch mode: queue leaf for deferred NN eval, return None.

    Returns the leaf value from root's perspective, or None if deferred.
    """
    if gamma_state.is_terminal():
      return self._terminal_value(gamma_state)

    gs = node.get_or_create_gamma(gamma_key)

    # First visit to this node by ANY γ — evaluate with NN
    if node.total_visits() == 0:
      if batch_mode:
        self._pending_leaves.append(
          (gamma_key, node, gamma_state, list(_path or []))
        )
        return None
      self._expand(gamma_key, gamma_state, node)
      gs.n += 1
      v = self._nn_value(gamma_state)
      return self._value_root_perspective(gamma_state, v)

    # First visit by THIS γ — expand its per-action stats
    if not gs.actions:
      self._expand(gamma_key, gamma_state, node)

    gs.n += 1

    action = self._selection(gamma_state, node)
    if action is None:
      # No legal action found – fall back to NN value
      return self._nn_value(gamma_state)

    # Descend
    gamma_state.apply_action(action)
    child_obs = self.obs_key(gamma_state, self.player_id)
    child_node = self.tree.get_or_create(
      child_obs, gamma_state.current_player()
    )
    child_gamma_key = gamma_state.serialize()
    child_node.get_or_create_gamma(child_gamma_key)

    # Build path for deferred backprop
    if batch_mode:
      child_path = list(_path or [])
      child_path.append((node, gamma_key, action))
    else:
      child_path = None

    v_root = self._search(
      child_gamma_key,
      gamma_state,
      child_node,
      batch_mode=batch_mode,
      _path=child_path,
    )

    if v_root is None:
      return None  # deferred

    # Backpropagation: sum-based W(γ, a)
    a_stat = gs.get_or_create_action(action)
    a_stat.n += 1
    a_stat.w += v_root
    return v_root

  # -- Batch NN evaluation --------------------------------------------------

  def _evaluate_pending_leaves(self) -> None:
    """Batch evaluate all pending leaf nodes and backpropagate values."""
    if not self._pending_leaves:
      return

    tensors = [
      self._state_tensor_side_to_move(state)
      for _, _, state, _ in self._pending_leaves
    ]
    batch = torch.stack(tensors, dim=0)

    with torch.no_grad():
      logits_batch, values_batch = self.net(batch)
      logits_batch = logits_batch.detach().cpu().numpy()
      values_batch = values_batch.detach().cpu().numpy()

    for i, (gamma_key, node, state, path) in enumerate(self._pending_leaves):
      # Expand the leaf if needed
      if not node.is_expanded:
        legal = list(state.legal_actions())
        node.is_expanded = True
        node.legal_actions = list(legal)

        logits = logits_batch[i]
        mask = np.full(
          (self.game.num_distinct_actions(),), -1e9, dtype=np.float32
        )
        for a in legal:
          mask[a] = 0.0
        priors = softmax.softmax_np(logits + mask)

        gs = node.get_or_create_gamma(gamma_key)
        for a in legal:
          a_stat = gs.get_or_create_action(a)
          a_stat.p = float(priors[a])
      else:
        gs = node.get_or_create_gamma(gamma_key)
        if not gs.actions:
          # Expand this γ's actions using batch logits
          legal = list(state.legal_actions())
          logits = logits_batch[i]
          mask = np.full(
            (self.game.num_distinct_actions(),), -1e9, dtype=np.float32
          )
          for a in legal:
            mask[a] = 0.0
          priors = softmax.softmax_np(logits + mask)
          for a in legal:
            a_stat = gs.get_or_create_action(a)
            a_stat.p = float(priors[a])

      gs.n += 1

      v = self._value_root_perspective(state, float(values_batch[i].item()))

      # Backpropagate along the stored path
      for ancestor, anc_gamma_key, action in path:
        anc_gs = ancestor.get_or_create_gamma(anc_gamma_key)
        a_stat = anc_gs.get_or_create_action(action)
        a_stat.n += 1
        a_stat.w += v

    self._pending_leaves.clear()

  # -- Visit-count policy vector --------------------------------------------

  def _root_visit_policy(
    self, root: tree.Node, temperature: float
  ) -> np.ndarray:
    """Build policy vector from root visit counts."""
    pi = np.zeros(self.game.num_distinct_actions(), dtype=np.float32)
    actions = root.legal_actions
    if not actions:
      return pi

    visits = np.array(
      [root.total_action_visits(a) for a in actions], dtype=np.float32
    )

    if temperature <= 1e-8:
      pi[actions[int(np.argmax(visits))]] = 1.0
      return pi

    vt = np.power(visits + 1e-8, 1.0 / float(temperature))
    probs = vt / float(np.sum(vt))
    for a, p in zip(actions, probs, strict=False):
      pi[a] = float(p)
    return pi

  # -- Override select_action: greedy from visit counts ---------------------

  def select_action(self, state: openspiel.State) -> int:
    """Select best action (greedy, no exploration noise)."""
    a, _ = self._select_action_impl(
      state, temperature=1e-8, add_dirichlet=False
    )
    return a

  def select_action_with_pi(
    self, state: openspiel.State
  ) -> tuple[int, np.ndarray]:
    """Select action with policy vector for training.

    Adds Dirichlet noise at root.  Uses temperature=1.0 for first 20
    plies, then greedy.
    """
    temperature = 1.0 if state.game_length() < 20 else 1e-8
    return self._select_action_impl(
      state, temperature=temperature, add_dirichlet=True
    )

  def _select_action_impl(
    self,
    state: openspiel.State,
    temperature: float,
    add_dirichlet: bool,
  ) -> tuple[int, np.ndarray]:
    """Core AZBSMCTS action selection: T×S loop with batched NN eval."""
    root_obs = self.obs_key(state, self.player_id)
    root = self.tree.get_or_create(root_obs, state.current_player())

    for _ in range(self.T):
      gamma_str, gamma_prior = self.sampler.sample_with_prior()

      gs = root.get_or_create_gamma(gamma_str)
      gs.prior = gamma_prior

      # Expand root for this γ with NN priors (+ optional Dirichlet)
      if not gs.actions:
        gamma_state = self.game.deserialize_state(gamma_str)
        self._expand(gamma_str, gamma_state, root, add_dirichlet=add_dirichlet)

      for _ in range(self.S):
        gamma_state = self.game.deserialize_state(gamma_str)
        self._search(gamma_str, gamma_state, root, batch_mode=True)

      self._evaluate_pending_leaves()

    pi = self._root_visit_policy(root, temperature=float(temperature))

    legal = state.legal_actions()
    probs = [pi[a] for a in legal]

    total = float(np.sum(probs))
    if total <= 0:
      best_a = root.get_most_visited_action(actions=legal)
      return int(best_a), pi

    probs /= total  # type: ignore
    r = self.rng.random()
    cum = 0.0
    action = 0
    for a in range(self.game.num_distinct_actions()):
      pa = float(probs[a])
      if pa <= 0.0:
        continue
      cum += pa
      if r <= cum:
        action = a
        break
    return int(action), pi
