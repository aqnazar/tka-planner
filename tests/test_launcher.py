"""The one-click launcher: it must not start a second server on a port already in use."""

import re
from pathlib import Path

LAUNCHER = Path(__file__).resolve().parents[1] / "Launch TKA Planner.cmd"


def test_the_launcher_refuses_a_port_something_already_listens_on():
    raw = LAUNCHER.read_bytes()
    text = raw.decode("ascii")
    guard = re.search(r'netstat -ano \| findstr /r /c:"[^"]*:%PORT% \.\*LISTENING" >nul '
                      r'&& \((.*?)\)', text, re.S)
    assert guard, "no LISTENING check on %PORT%"
    assert "exit /b 1" in guard.group(1)
    # Checked before anything is started: the browser waiter and the server.
    assert guard.start() < text.index("start \"\" /b powershell")
    assert guard.start() < text.index("-m tka_planner.server")
    # cmd.exe reads labels and blocks reliably only with CRLF line ends.
    assert raw.count(b"\n") == raw.count(b"\r\n")
