"""Which Fusion documents a build copies, and where the add-in listens.

The documents are named by their Fusion lineage identifiers, which belong to the
author's Fusion account and stay out of the repository: they are read from
``fusion/local.json`` (git-ignored), laid out as ``fusion/local.example.json``, or from
the file ``TKA_FUSION_CONFIG`` names.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from .transport import DEFAULT_URL, address_of

__all__ = ["FusionConfig", "SOURCE_PARTS", "default_config_path"]

# The five parametric designs a knee's build copies.
SOURCE_PARTS = ("femoral", "plate", "insert", "block", "tblock")
REPO = Path(__file__).resolve().parents[2]


def default_config_path() -> Path:
    return Path(os.environ.get("TKA_FUSION_CONFIG") or REPO / "fusion" / "local.json")


@dataclass
class FusionConfig:
    url: str = DEFAULT_URL
    sources: dict = field(default_factory=dict)
    path: Path | None = None

    @property
    def address(self) -> str:
        return address_of(self.url)

    @classmethod
    def load(cls, path=None) -> "FusionConfig":
        """The configuration in ``path`` (default :func:`default_config_path`); an empty
        one, with the default address, when the file is not there."""
        path = Path(path) if path else default_config_path()
        if not path.is_file():
            return cls(path=path)
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(url=data.get("mcp_url") or DEFAULT_URL,
                   sources=dict(data.get("sources") or {}), path=path)

    def require(self) -> "FusionConfig":
        """Itself, or an error saying what is missing and where to put it."""
        missing = [p for p in SOURCE_PARTS if not self.sources.get(p)
                   or "<" in str(self.sources[p])]
        if missing:
            where = self.path or default_config_path()
            raise FileNotFoundError(
                f"No Fusion source documents for {', '.join(missing)}: copy "
                f"fusion/local.example.json to {where} and fill in the designs' lineage "
                f"identifiers (kept out of the repository).")
        return self
