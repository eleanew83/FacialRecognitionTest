"""I/O utility function placeholders."""

from typing import Any

def save_json(path: str, payload: dict[str, Any]) -> None:
    """Persist a JSON payload to disk."""
    ...


def load_json(path: str) -> dict[str, Any]:
    """Load JSON metadata from disk."""
    ...
