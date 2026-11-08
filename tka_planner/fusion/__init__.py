"""Fusion 360 from the planner: the MCP transport and the build of a knee's parts.

Nothing here runs without Fusion open and its MCP add-in listening (127.0.0.1:27182);
everything else in the planner works without it. Fusion file identifiers come from the
local, git-ignored ``fusion/local.json``.
"""

from .build import BuildResult, build_case
from .config import FusionConfig
from .transport import FusionError, FusionUnavailable, call_fusion, ping

__all__ = ["BuildResult", "build_case", "FusionConfig", "FusionError", "FusionUnavailable",
           "call_fusion", "ping"]
