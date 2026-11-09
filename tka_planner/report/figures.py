"""Figures for the HTML report, drawn without a plotting library.

The report must open from a memory stick in ten years' time, so a figure is a PNG
written into the page as a data URI. The fit map is already an image -- one value per
cell of a regular grid in the cut plane -- so it needs no plotting library at all: each
cell is a pixel, coloured by the same scale the viewer draws, and the PNG is written
here with ``zlib`` from the standard library.
"""

from __future__ import annotations

import base64
import html
import struct
import zlib

import numpy as np

from tka_planner.scene.fitmap import SCALE_MM, STOPS, colours_for

__all__ = ["png_bytes", "fit_map_figure", "legend_bar"]


def png_bytes(rgba: np.ndarray) -> bytes:
    """Encode an ``(H, W, 4)`` array of 0-255 values as a PNG."""
    rgba = np.ascontiguousarray(np.clip(rgba, 0, 255).astype(np.uint8))
    height, width = rgba.shape[:2]
    raw = b"".join(b"\x00" + rgba[row].tobytes() for row in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))

    header = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def fit_map_image(grid) -> np.ndarray:
    """The fit grid as RGBA pixels: anterior up, lateral to the right, one per cell."""
    values = grid.signed_mm[::-1]  # rows run posterior to anterior; images run down
    rgba = np.zeros(values.shape + (4,))
    valid = np.isfinite(values)
    rgba[valid, :3] = colours_for(values[valid]) * 255.0
    rgba[valid, 3] = 255.0
    return rgba


def legend_bar() -> str:
    """The colour scale as a CSS gradient, with its ends and its middle named."""
    span = 2.0 * SCALE_MM
    stops = ", ".join(
        f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)}) "
        f"{100.0 * (value + SCALE_MM) / span:.1f}%"
        for value, (r, g, b) in STOPS)
    return (
        '<div class="legend"><div class="bar" style="background:linear-gradient('
        f'to right, {stops})"></div>'
        f'<div class="ends"><span>&minus;{SCALE_MM:g} mm overhang</span>'
        '<span>0 seated</span>'
        f'<span>+{SCALE_MM:g} mm bone uncovered</span></div></div>'
    )


def fit_map_figure(grid, *, title: str, record: dict | None = None) -> str:
    """One cut's fit map as an inline figure, with the numbers it shows beneath it."""
    image = fit_map_image(grid)
    data = base64.b64encode(png_bytes(image)).decode("ascii")
    cell = float(abs(grid.xs[1] - grid.xs[0])) if len(grid.xs) > 1 else 0.25
    width_mm = image.shape[1] * cell
    caption = [f"{html.escape(title)}: {width_mm:.0f} mm across, one pixel per "
               f"{cell:g} mm cell. Anterior is up and lateral to the right."]
    if record:
        caption.append(
            f"Coverage {100 * record['coverage_fraction']:.1f}%, overhang up to "
            f"{record['max_overhang_mm']:.1f} mm, bone uncovered up to "
            f"{record['max_underhang_mm']:.1f} mm.")
    return (
        '<figure class="fitmap">'
        f'<img alt="{html.escape(title)}" src="data:image/png;base64,{data}" '
        f'width="{2 * image.shape[1]}" height="{2 * image.shape[0]}">'
        f'<figcaption>{" ".join(caption)}</figcaption></figure>'
    )
