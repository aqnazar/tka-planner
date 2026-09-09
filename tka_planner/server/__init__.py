"""The local application: routes, sessions, and an HTTP adapter in front of them.

The engine is driven the same way from here as it is from a test. What this package
adds is a way for a browser to do the driving, and a store behind the commits so that
returning to a plan already cut costs a disk read rather than a boolean.
"""

from .api import Api, ApiError, Binary
from .httpd import PlannerServer
from .sessions import SessionEntry, SessionManager

__all__ = [
    "Api", "ApiError", "Binary", "PlannerServer", "SessionManager", "SessionEntry",
]
