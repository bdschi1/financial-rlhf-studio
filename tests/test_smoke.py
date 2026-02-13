"""Smoke tests for financial-rlhf-studio."""
import os
def test_app_imports():
    """Verify app.py imports without error."""
    import app


def test_src_directory_exists():
    """Verify src/ module is present."""
    src_dir = os.path.join(os.path.dirname(__file__), "..", "src")
    assert os.path.isdir(src_dir), "src/ directory missing"


def test_data_directory_exists():
    """Verify data/ directory is present."""
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    assert os.path.isdir(data_dir), "data/ directory missing"


def test_data_card_exists():
    """Verify DATA_CARD.md is present (dataset documentation)."""
    card_path = os.path.join(os.path.dirname(__file__), "..", "DATA_CARD.md")
    assert os.path.isfile(card_path), "DATA_CARD.md missing"
