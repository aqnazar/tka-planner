"""Capture a key moment as a paper-figure candidate.

  python shoot.py <page.html|image.svg> <slug> [query] [width] [height]

Writes ../figures/<YYYY-MM-DD>_<NN>_<slug>.png with headless Chrome (NN counts up per
day). Then add a row to ../figures/FIGURES.md and a line in LAB_JOURNAL.md.
Private: figures built from patient scans stay in this gitignored folder.
"""

import datetime
import subprocess
import sys
from pathlib import Path

FIGURES = Path(__file__).resolve().parents[1] / "figures"
CHROME = [Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
          Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe")]


def main():
    src, slug = Path(sys.argv[1]).resolve(), sys.argv[2]
    query = sys.argv[3] if len(sys.argv) > 3 else ""
    width = sys.argv[4] if len(sys.argv) > 4 else "1500"
    height = sys.argv[5] if len(sys.argv) > 5 else "1400"
    day = datetime.date.today().isoformat()
    FIGURES.mkdir(exist_ok=True)
    n = len(list(FIGURES.glob(f"{day}_*.png"))) + 1
    out = FIGURES / f"{day}_{n:02d}_{slug}.png"
    chrome = next(c for c in CHROME if c.exists())
    url = src.as_uri() + (("?" + query.lstrip("?")) if query else "")
    subprocess.run([str(chrome), "--headless=new", "--disable-gpu", "--hide-scrollbars",
                    f"--window-size={width},{height}", "--virtual-time-budget=4000",
                    f"--screenshot={out}", url], check=True, capture_output=True)
    print(out)


if __name__ == "__main__":
    main()
