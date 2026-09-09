"""``python -m tka_planner.server`` -- start the planner and open it."""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

from .httpd import PlannerServer
from .sessions import SessionManager

DEFAULT_STORE = Path.home() / ".tka-planner"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tka serve",
        description=(
            "Run the planner as a local application. Research and demonstration "
            "only; not a medical device."
        ),
    )
    parser.add_argument(
        "--cases", default=None,
        help="folder holding one subfolder per case; each is listed if it holds a "
             "femur and a tibia",
    )
    parser.add_argument("--library", default=None, help="implant library folder")
    parser.add_argument(
        "--store", default=str(DEFAULT_STORE),
        help="where cases, plans and committed geometry are kept",
    )
    parser.add_argument("--port", type=int, default=8731)
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="interface to bind; the default keeps patient geometry off the network",
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="do not open a browser window"
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    manager = SessionManager(
        store=args.store, cases_root=args.cases, library=args.library
    )
    server = PlannerServer(manager, host=args.host, port=args.port)

    print(f"TKA planner running at {server.url}")
    print(f"  store    {args.store}")
    print(f"  cases    {args.cases or '(none configured; open a folder by path)'}")
    print(f"  library  {args.library or '(none configured; implants will not show)'}")
    print("Press Ctrl+C to stop.")

    if not args.no_browser:
        webbrowser.open(server.url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        server.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
