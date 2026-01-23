"""Tests for belief.tree module."""

import pytest

from src.belief.tree import BeliefTree, EdgeStats, Node


class TestEdgeStats:
    """Test EdgeStats dataclass."""

    def test_default_values(self):
        """Test default values for EdgeStats."""
        edge = EdgeStats()
        assert edge.n == 0
        assert edge.w == 0.0
        assert edge.p == 0.0

    def test_q_property_zero_visits(self):
        """Test q property returns 0 when n is 0."""
        edge = EdgeStats(n=0, w=10.0)
        assert edge.q == 0.0

    def test_q_property_with_visits(self):
        """Test q property calculates average correctly."""
        edge = EdgeStats(n=5, w=10.0)
        assert edge.q == 2.0

    def test_negative_values(self):
        """Test q property with negative total value."""
        edge = EdgeStats(n=4, w=-8.0)
        assert edge.q == -2.0


class TestNode:
    """Test Node class."""

    def test_node_initialization(self):
        """Test node is initialized with correct defaults."""
        node = Node(obs_key="test_obs", player_to_act=0)
        assert node.obs_key == "test_obs"
        assert node.player_to_act == 0
        assert node.is_expanded is False
        assert len(node.edges) == 0
        assert len(node.legal_actions) == 0
        assert node.n == 0

    def test_get_most_visited_action_basic(self):
        """Test get_most_visited_action returns action with highest visit count."""
        node = Node(obs_key="test", player_to_act=0)
        node.edges[0] = EdgeStats(n=5)
        node.edges[1] = EdgeStats(n=10)
        node.edges[2] = EdgeStats(n=3)

        assert node.get_most_visited_action() == 1

    def test_get_most_visited_action_filtered(self):
        """Test get_most_visited_action with filtered action list."""
        node = Node(obs_key="test", player_to_act=0)
        node.edges[0] = EdgeStats(n=5)
        node.edges[1] = EdgeStats(n=10)
        node.edges[2] = EdgeStats(n=3)

        # Only consider actions 0 and 2
        assert node.get_most_visited_action(actions=[0, 2]) == 0

    def test_get_most_visited_action_keyerror_bug(self):
        """Test handling actions not in edges (uses default EdgeStats with n=0).

        This reproduces the scenario from the eval script where an action
        from the legal actions list is not present in node.edges.
        Actions not in edges are treated as having 0 visits.
        """
        node = Node(obs_key="test", player_to_act=0)
        node.edges[1] = EdgeStats(n=10)
        node.edges[2] = EdgeStats(n=3)

        # Action 0 is in the actions list but not in edges
        # Should treat it as n=0 and return 1 (highest visits)
        assert node.get_most_visited_action(actions=[0, 1, 2]) == 1

    def test_get_most_visited_action_all_zero_visits(self):
        """Test when all actions have zero visits (not in edges)."""
        node = Node(obs_key="test", player_to_act=0)
        node.edges[5] = EdgeStats(n=10)  # Not in the actions list

        # All actions [0,1,2] have 0 visits (not in edges)
        # Should return first one (0) since they're all equal
        result = node.get_most_visited_action(actions=[0, 1, 2])
        assert result in [0, 1, 2]  # Any is valid since all have n=0

    def test_get_most_visited_action_empty_edges(self):
        """Test get_most_visited_action with no edges."""
        node = Node(obs_key="test", player_to_act=0)

        # Should raise ValueError from max() on empty sequence
        with pytest.raises(ValueError):
            node.get_most_visited_action()


class TestBeliefTree:
    """Test BeliefTree class."""

    def test_initialization(self):
        """Test belief tree initialization."""
        tree = BeliefTree()
        assert len(tree.nodes) == 0

    def test_get_or_create_new_node(self):
        """Test creating a new node."""
        tree = BeliefTree()
        node = tree.get_or_create("obs1", player_to_act=0)

        assert node.obs_key == "obs1"
        assert node.player_to_act == 0
        assert len(tree.nodes) == 1
        assert "obs1" in tree.nodes

    def test_get_or_create_existing_node(self):
        """Test retrieving an existing node."""
        tree = BeliefTree()
        node1 = tree.get_or_create("obs1", player_to_act=0)
        node1.n = 5  # Modify the node

        node2 = tree.get_or_create("obs1", player_to_act=0)

        # Should return the same node
        assert node2 is node1
        assert node2.n == 5
        assert len(tree.nodes) == 1

    def test_multiple_nodes(self):
        """Test creating multiple nodes."""
        tree = BeliefTree()
        node1 = tree.get_or_create("obs1", player_to_act=0)
        node2 = tree.get_or_create("obs2", player_to_act=1)

        assert len(tree.nodes) == 2
        assert node1 is not node2
        assert tree.nodes["obs1"] is node1
        assert tree.nodes["obs2"] is node2
