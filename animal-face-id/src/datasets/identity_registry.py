"""Canonical macaque names and current groups, independent of capture locations."""
from __future__ import annotations

import json
from pathlib import Path


class IdentityRegistry:
    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        self.payload = json.loads(self.path.read_text())
        self.identities = self.payload["identities"]
        self.aliases = self.payload["aliases"]
        for name, info in self.identities.items():
            if Path(info["source_directory"]).parts != (info["current_group"], info["gender"], name):
                raise ValueError(f"Invalid canonical source directory for {name}")
        if any(name not in self.identities for name in self.aliases.values()):
            raise ValueError("Identity alias points to an unknown individual")

    def resolve(self, name: str) -> str:
        if name in self.identities:
            return name
        try:
            return self.aliases[name]
        except KeyError as error:
            raise ValueError(f"Unknown macaque identity: {name!r}; update the source registry first") from error

    def metadata(self, name: str) -> dict:
        canonical = self.resolve(name)
        return {"id": canonical, **self.identities[canonical],
                "identity_registry_version": self.payload["version"]}

    def validate(self, manifest: dict) -> None:
        for rows in manifest.values():
            for row in rows:
                info = self.metadata(row["id"])
                for field in ("id", "current_group", "gender"):
                    if row.get(field) != info[field]:
                        raise ValueError(f"Noncanonical identity/group in {row['path']}: {field}")


def discover_registry(source: str | Path, explicit: str | Path | None = None):
    """Generated datasets carry a registry; generic test/other-species inputs do not."""
    if explicit is not None:
        return IdentityRegistry(explicit)
    source = Path(source).resolve()
    for parent in (source, source.parent):
        path = parent / "identity_registry.json"
        if path.is_file():
            return IdentityRegistry(path)
    return None


def unique_destination(relative: Path, digest: str, occupied: set[str]) -> Path:
    """Never overwrite a different photo when two old identities merge."""
    candidate = relative
    if str(candidate) in occupied:
        candidate = relative.with_name(f"{relative.stem}__{digest[:16]}{relative.suffix}")
    if str(candidate) in occupied:
        raise ValueError(f"Repeated destination/content needs review: {relative}")
    occupied.add(str(candidate))
    return candidate
