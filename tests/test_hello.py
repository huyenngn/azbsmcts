"""Basic sanity test to verify pytest is working."""


def test_hello_world():
    """A simple test to verify pytest setup."""
    assert True


def test_basic_math():
    """Test basic arithmetic."""
    assert 1 + 1 == 2
    assert 2 * 3 == 6


def test_string_operations():
    """Test string operations."""
    assert "hello".upper() == "HELLO"
    assert len("test") == 4
