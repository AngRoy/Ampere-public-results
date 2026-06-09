from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """Return the repository root from the installed source layout."""

    return Path(__file__).resolve().parents[3]


def as_repo_relative(path: str | Path, root: str | Path | None = None) -> str:
    base = Path(root).resolve() if root is not None else repo_root()
    return str(Path(path).resolve().relative_to(base)).replace("\\", "/")
