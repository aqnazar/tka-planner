"""Talking to Fusion 360: a script sent to its MCP add-in, the reply read back.

The add-in listens on 127.0.0.1:27182 and speaks MCP over HTTP (JSON-RPC, replies as an
event stream). A script is Python that runs inside Fusion with the adsk API. From
``eval/fusion_driver.call_fusion``, with one change: Fusion not running, or its add-in
not started, is a :class:`FusionUnavailable` that names the address, not a socket error.
"""

from __future__ import annotations

import json
import socket
import urllib.error
import urllib.parse
import urllib.request

__all__ = ["DEFAULT_URL", "FusionUnavailable", "FusionError", "call_fusion", "ping",
           "address_of"]

DEFAULT_URL = "http://127.0.0.1:27182/mcp"
HEADERS = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}


class FusionUnavailable(RuntimeError):
    """Fusion, or its MCP add-in, is not answering."""


class FusionError(RuntimeError):
    """Fusion answered, and the script failed."""


def address_of(url: str) -> str:
    parts = urllib.parse.urlsplit(url)
    return f"{parts.hostname}:{parts.port}"


def _unavailable(url, error) -> FusionUnavailable:
    reason = getattr(error, "reason", error)
    return FusionUnavailable(
        f"Fusion is not reachable on {address_of(url)} ({reason}). Open Fusion 360 and "
        f"start its MCP add-in, then build again.")


def _post(url, body, sid=None, timeout=900):
    headers = dict(HEADERS, **({"Mcp-Session-Id": sid} if sid else {}))
    request = urllib.request.Request(url, json.dumps(body).encode(), headers)
    try:
        reply = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as error:          # it answered: not "unreachable"
        raise FusionError(f"Fusion's add-in answered HTTP {error.code}: {error.reason}") from None
    except (urllib.error.URLError, ConnectionError, socket.timeout, OSError) as error:
        raise _unavailable(url, error) from None
    return reply.headers.get("Mcp-Session-Id"), reply.read().decode("utf-8")


def _initialize(url, timeout):
    sid, _ = _post(url, {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                         "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                                    "clientInfo": {"name": "tka", "version": "0"}}},
                   timeout=timeout)
    _post(url, {"jsonrpc": "2.0", "method": "notifications/initialized"}, sid, timeout=timeout)
    return sid


def ping(url: str = DEFAULT_URL, timeout: float = 5) -> None:
    """Raise :class:`FusionUnavailable` unless the add-in answers an initialize."""
    _initialize(url, timeout)


def call_fusion(script: str, read_only: bool, url: str = DEFAULT_URL,
                timeout: float = 900) -> str:
    """Run ``script`` in Fusion and return what it printed (the message of the reply)."""
    sid = _initialize(url, timeout)
    _, out = _post(url, {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                         "params": {"name": "fusion_mcp_execute",
                                    "arguments": {"featureType": "script",
                                                  "object": {"readOnly": read_only,
                                                             "script": script}}}},
                   sid, timeout=timeout)
    lines = [ln[5:] for ln in out.splitlines() if ln.startswith("data:")] or [out]
    res = json.loads(lines[-1])
    if "error" in res:
        raise FusionError(str(res["error"]))
    text = "".join(c.get("text", "") for c in res["result"]["content"])
    try:
        reply = json.loads(text)
    except ValueError:
        return text
    if isinstance(reply, dict):
        if reply.get("error") or reply.get("success") is False:
            raise FusionError(f"Fusion: {reply.get('error')} | {reply.get('message', '')[-2000:]}")
        return reply.get("message", text)
    return text
