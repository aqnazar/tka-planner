"""Package the add-on into a zip Blender's "Install from Disk" accepts.

Run with plain Python::

    python build_addon.py

Then in Blender: **Edit > Preferences > Add-ons**, open the dropdown at the top right,
choose **Install from Disk...**, and select the generated ``tka_planner.zip``.

Adding the repository to Blender's script paths does not work, and it is worth saying
why: Blender scans a script path for an ``addons`` subdirectory, and inside that expects
either a module or a package whose ``__init__`` declares ``bl_info``. A repository
checkout matches neither shape, so nothing appears in the add-on list no matter how many
paths are added or how often the list is refreshed.

The zip carries the whole package, including the planning core, so the add-on has no
dependency on where this repository happens to live. The one thing it does not carry is
the size chart, which is read from the path recorded in the add-on's preferences.
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parent
PACKAGE = REPOSITORY / "tka_planner"
OUTPUT = REPOSITORY / "tka_planner.zip"

# Kept out of the zip: caches, and anything that is not source or data.
EXCLUDED_DIRECTORIES = {"__pycache__", ".pytest_cache"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


def build(output: Path = OUTPUT) -> Path:
    if not PACKAGE.is_dir():
        raise FileNotFoundError(f"Package not found: {PACKAGE}")

    if output.exists():
        output.unlink()

    included = 0
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(PACKAGE.rglob("*")):
            if path.is_dir():
                continue
            if any(part in EXCLUDED_DIRECTORIES for part in path.parts):
                continue
            if path.suffix in EXCLUDED_SUFFIXES:
                continue
            archive.write(path, path.relative_to(REPOSITORY))
            included += 1

        # The size chart travels with the add-on so a fresh install can plan without
        # being pointed at the repository first.
        chart = REPOSITORY / "data" / "SizeChart.csv"
        if chart.is_file():
            archive.write(chart, Path("tka_planner") / "data" / "SizeChart.csv")
            included += 1

    print(f"Wrote {output.name} ({included} files, {output.stat().st_size:,} bytes)")
    print()
    print("In Blender: Edit > Preferences > Add-ons > (dropdown) Install from Disk...")
    print(f"then select {output}")
    return output


if __name__ == "__main__":
    build()
