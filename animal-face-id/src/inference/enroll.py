"""Unknown clustering and enrollment entry points."""

from typing import Any

def cluster_unknowns(unknown_pool_path: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    """Cluster unknown embeddings and return candidate groups."""
    ...


def enroll_new_id(cluster_summary: dict[str, Any], gallery_path: str) -> None:
    """Add a newly approved cluster centroid into the gallery store."""
    ...
