#!/usr/bin/env python3
"""Render an SVG figure to PNG using the first available local renderer.

Supported backends:
- inkscape
- rsvg-convert
- magick
- convert
- cairosvg (Python package)
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def render_with_cli(svg: Path, png: Path, width: int, height: int) -> str | None:
    if shutil.which("rsvg-convert"):
        run(
            [
                "rsvg-convert",
                "-w",
                str(width),
                "-h",
                str(height),
                str(svg),
                "-o",
                str(png),
            ]
        )
        return "rsvg-convert"

    if shutil.which("inkscape"):
        run(
            [
                "inkscape",
                str(svg),
                "--export-type=png",
                f"--export-filename={png}",
                "-w",
                str(width),
                "-h",
                str(height),
            ]
        )
        return "inkscape"

    if shutil.which("magick"):
        run(
            [
                "magick",
                "-background",
                "white",
                "-density",
                "200",
                str(svg),
                "-resize",
                f"{width}x{height}",
                str(png),
            ]
        )
        return "magick"

    if shutil.which("convert"):
        run(
            [
                "convert",
                "-background",
                "white",
                "-density",
                "200",
                str(svg),
                "-resize",
                f"{width}x{height}",
                str(png),
            ]
        )
        return "convert"

    return None


def render_with_cairosvg(svg: Path, png: Path, width: int, height: int) -> str | None:
    try:
        import cairosvg  # type: ignore
    except ModuleNotFoundError:
        return None

    cairosvg.svg2png(
        url=str(svg),
        write_to=str(png),
        output_width=width,
        output_height=height,
    )
    return "cairosvg"


def main() -> int:
    parser = argparse.ArgumentParser(description="Render an SVG figure to PNG.")
    parser.add_argument("svg", type=Path, help="Input SVG path")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output PNG path (default: same name as SVG with .png suffix)",
    )
    parser.add_argument("--width", type=int, default=1600, help="Output width in pixels")
    parser.add_argument("--height", type=int, default=900, help="Output height in pixels")
    args = parser.parse_args()

    svg = args.svg
    png = args.output or svg.with_suffix(".png")

    if not svg.exists():
        print(f"Input SVG not found: {svg}", file=sys.stderr)
        return 1

    png.parent.mkdir(parents=True, exist_ok=True)

    backend = render_with_cli(svg, png, args.width, args.height)
    if backend is None:
        backend = render_with_cairosvg(svg, png, args.width, args.height)

    if backend is None:
        print("No SVG-to-PNG renderer found.", file=sys.stderr)
        print("Install one of: inkscape, librsvg2-bin (rsvg-convert), imagemagick, or cairosvg.", file=sys.stderr)
        return 1

    print(f"Rendered {png} using {backend}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
