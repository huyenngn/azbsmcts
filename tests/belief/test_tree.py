"""Tests for belief.tree module (BS-MCTS data structures)."""

import pytest

from belief import tree

# ---------------------------------------------------------------------------
# GammaActionStats
# ---------------------------------------------------------------------------


class TestGammaActionStats:
  """Test per-(γ, action) statistics."""

  def test_defaults(self) -> None:
    s = tree.GammaActionStats()
    assert s.n == 0
    assert s.u == 0.0

  def test_running_average_update(self) -> None:
    """Simulate Eq 2: U(γ,a) ← U(γ,a) + (R − U(γ,a)) / N(γ,a)."""
    s = tree.GammaActionStats()
    rewards = [1.0, 0.0, 0.5]
    for r in rewards:
      s.n += 1
      s.u += (r - s.u) / s.n
    assert s.n == 3
    assert s.u == pytest.approx(0.5, abs=1e-9)


# ---------------------------------------------------------------------------
# GammaStats
# ---------------------------------------------------------------------------


class TestGammaStats:
  """Test per-determinization statistics."""

  def test_defaults(self) -> None:
    gs = tree.GammaStats()
    assert gs.n == 0
    assert gs.prior == 1.0
    assert gs.actions == {}

  def test_expected_utility_no_actions(self) -> None:
    gs = tree.GammaStats()
    assert gs.expected_utility() == 0.0

  def test_expected_utility(self) -> None:
    """U(γ) = Σ U(γ,a)·N(γ,a) / Σ N(γ,a) (Eq 5)."""
    gs = tree.GammaStats()
    gs.actions[0] = tree.GammaActionStats(n=2, u=1.0)  # contributes 2*1.0
    gs.actions[1] = tree.GammaActionStats(n=3, u=0.5)  # contributes 3*0.5
    # U(γ) = (2*1.0 + 3*0.5) / (2 + 3) = 3.5 / 5 = 0.7
    assert gs.expected_utility() == pytest.approx(0.7)

  def test_get_or_create_action(self) -> None:
    gs = tree.GammaStats()
    a = gs.get_or_create_action(42)
    assert a.n == 0
    assert a.u == 0.0
    # Same object returned on second call
    assert gs.get_or_create_action(42) is a


# ---------------------------------------------------------------------------
# EdgeStats (AZMCTS compat)
# ---------------------------------------------------------------------------


class TestEdgeStats:
  """Test EdgeStats dataclass (kept for AZMCTS)."""

  def test_default_values(self) -> None:
    edge = tree.EdgeStats()
    assert edge.n == 0
    assert edge.w == 0.0
    assert edge.p == 0.0

  def test_q_property_zero_visits(self) -> None:
    edge = tree.EdgeStats(n=0, w=10.0)
    assert edge.q == 0.0

  def test_q_property_with_visits(self) -> None:
    edge = tree.EdgeStats(n=5, w=10.0)
    assert edge.q == 2.0

  def test_negative_values(self) -> None:
    edge = tree.EdgeStats(n=4, w=-8.0)
    assert edge.q == -2.0


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------


class TestNode:
  """Test belief-state Node."""

  def test_initialization(self) -> None:
    node = tree.Node(obs_key="test_obs", player_to_act=0)
    assert node.obs_key == "test_obs"
    assert node.player_to_act == 0
    assert node.is_expanded is False
    assert len(node.edges) == 0
    assert len(node.gammas) == 0
    assert node.n == 0

  # -- per-γ helpers --------------------------------------------------------

  def test_get_or_create_gamma(self) -> None:
    node = tree.Node(obs_key="x", player_to_act=0)
    gs = node.get_or_create_gamma("g1")
    assert gs.n == 0
    assert node.get_or_create_gamma("g1") is gs
    assert len(node.gammas) == 1

  # -- belief weights (Eq 4) -----------------------------------------------

  def test_belief_weights_empty(self) -> None:
    node = tree.Node(obs_key="x", player_to_act=0)
    assert node.belief_weights() == {}

  def test_belief_weights_uniform_when_no_data(self) -> None:
    """With no action stats, all U(γ)=0 → uniform beliefs."""
    node = tree.Node(obs_key="x", player_to_act=0)
    node.get_or_create_gamma("g1")
    node.get_or_create_gamma("g2")
    bw = node.belief_weights()
    assert bw["g1"] == pytest.approx(0.5)
    assert bw["g2"] == pytest.approx(0.5)

  def test_belief_weights_sharp_with_high_lambda(self) -> None:
    """Higher λ → sharper beliefs toward the γ with higher U(γ)."""
    node = tree.Node(obs_key="x", player_to_act=0)
    gs1 = node.get_or_create_gamma("g1")
    gs1.actions[0] = tree.GammaActionStats(n=5, u=1.0)  # U(γ1) = 1.0
    gs2 = node.get_or_create_gamma("g2")
    gs2.actions[0] = tree.GammaActionStats(n=5, u=0.0)  # U(γ2) = 0.0

    bw = node.belief_weights(lambda_guess=10.0)
    assert bw["g1"] > bw["g2"]

  # -- belief-weighted U(B,a) (Eq 3) ---------------------------------------

  def test_belief_weighted_u_empty(self) -> None:
    node = tree.Node(obs_key="x", player_to_act=0)
    assert node.belief_weighted_u(0) == 0.0

  def test_belief_weighted_u_single_gamma(self) -> None:
    node = tree.Node(obs_key="x", player_to_act=0)
    gs = node.get_or_create_gamma("g1")
    gs.actions[0] = tree.GammaActionStats(n=3, u=0.8)
    # Only one γ → b_1 = 1.0, so U(B, 0) = 0.8
    assert node.belief_weighted_u(0) == pytest.approx(0.8)

  def test_belief_weighted_u_two_gammas(self) -> None:
    """U(B,a) = Σ b_i · U(γ_i, a)."""
    node = tree.Node(obs_key="x", player_to_act=0)
    # Both γ have same expected utility → uniform beliefs
    gs1 = node.get_or_create_gamma("g1")
    gs1.actions[0] = tree.GammaActionStats(n=2, u=1.0)
    gs2 = node.get_or_create_gamma("g2")
    gs2.actions[0] = tree.GammaActionStats(n=2, u=0.0)
    # With λ=0 (force uniform): U(B,0) = 0.5*1.0 + 0.5*0.0 = 0.5
    # λ=0 → all logits=0 → uniform
    u = node.belief_weighted_u(0, lambda_guess=0.0)
    assert u == pytest.approx(0.5)

  # -- aggregate visits -----------------------------------------------------

  def test_total_action_visits(self) -> None:
    node = tree.Node(obs_key="x", player_to_act=0)
    gs1 = node.get_or_create_gamma("g1")
    gs1.actions[0] = tree.GammaActionStats(n=3)
    gs2 = node.get_or_create_gamma("g2")
    gs2.actions[0] = tree.GammaActionStats(n=5)
    assert node.total_action_visits(0) == 8
    assert node.total_action_visits(1) == 0

  def test_total_visits(self) -> None:
    node = tree.Node(obs_key="x", player_to_act=0)
    node.get_or_create_gamma("g1").n = 10
    node.get_or_create_gamma("g2").n = 7
    assert node.total_visits() == 17

  # -- UCT value (Eq 1) ----------------------------------------------------

  def test_uct_value_explores_unvisited(self) -> None:
    """Action with 0 visits should have high UCT score (exploration)."""
    node = tree.Node(obs_key="x", player_to_act=0)
    gs = node.get_or_create_gamma("g1")
    gs.n = 10
    gs.actions[0] = tree.GammaActionStats(n=8, u=0.5)
    gs.actions[1] = tree.GammaActionStats(n=0, u=0.0)

    v0 = node.uct_value(0, c_uct=1.4)
    v1 = node.uct_value(1, c_uct=1.4)
    assert v1 > v0  # Unvisited action should be preferred

  # -- Opponent Predicting (Eq 6) ------------------------------------------

  def test_opponent_action_probs_uniform_no_data(self) -> None:
    """Without data, all actions should be ~uniform."""
    node = tree.Node(obs_key="x", player_to_act=1)
    node.get_or_create_gamma("g1")
    probs = node.opponent_action_probs([0, 1, 2])
    assert len(probs) == 3
    assert sum(probs) == pytest.approx(1.0)
    # All should be roughly equal since U(B,a)=0 for all
    assert probs[0] == pytest.approx(probs[1], abs=1e-6)

  def test_opponent_action_probs_empty(self) -> None:
    node = tree.Node(obs_key="x", player_to_act=1)
    assert node.opponent_action_probs([]) == []

  # -- legacy get_most_visited_action (AZMCTS compat) --------------------

  def test_get_most_visited_action_basic(self) -> None:
    node = tree.Node(obs_key="test", player_to_act=0)
    node.edges[0] = tree.EdgeStats(n=5)
    node.edges[1] = tree.EdgeStats(n=10)
    node.edges[2] = tree.EdgeStats(n=3)
    assert node.get_most_visited_action() == 1

  def test_get_most_visited_action_filtered(self) -> None:
    node = tree.Node(obs_key="test", player_to_act=0)
    node.edges[0] = tree.EdgeStats(n=5)
    node.edges[1] = tree.EdgeStats(n=10)
    node.edges[2] = tree.EdgeStats(n=3)
    assert node.get_most_visited_action(actions=[0, 2]) == 0

  def test_get_most_visited_action_missing_key(self) -> None:
    node = tree.Node(obs_key="test", player_to_act=0)
    node.edges[1] = tree.EdgeStats(n=10)
    assert node.get_most_visited_action(actions=[0, 1, 2]) == 1

  def test_get_most_visited_action_empty_edges(self) -> None:
    node = tree.Node(obs_key="test", player_to_act=0)
    with pytest.raises(ValueError):
      node.get_most_visited_action()


# ---------------------------------------------------------------------------
# BeliefTree
# ---------------------------------------------------------------------------


class TestBeliefTree:
  """Test BeliefTree class."""

  def test_initialization(self) -> None:
    bt = tree.BeliefTree()
    assert len(bt._nodes) == 0

  def test_get_or_create_new_node(self) -> None:
    bt = tree.BeliefTree()
    node = bt.get_or_create("obs1", player_to_act=0)
    assert node.obs_key == "obs1"
    assert len(bt._nodes) == 1

  def test_get_or_create_existing_node(self) -> None:
    bt = tree.BeliefTree()
    n1 = bt.get_or_create("obs1", player_to_act=0)
    n1.n = 5
    n2 = bt.get_or_create("obs1", player_to_act=0)
    assert n2 is n1
    assert n2.n == 5

  def test_multiple_nodes(self) -> None:
    bt = tree.BeliefTree()
    n1 = bt.get_or_create("obs1", player_to_act=0)
    n2 = bt.get_or_create("obs2", player_to_act=1)
    assert len(bt._nodes) == 2
    assert n1 is not n2

  def test_clear(self) -> None:
    bt = tree.BeliefTree()
    bt.get_or_create("obs1", player_to_act=0)
    bt.clear()
    assert len(bt._nodes) == 0
