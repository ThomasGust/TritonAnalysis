"""Placeholder for the removed crab video detector."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    """Build the placeholder parser for the removed batch video detector."""
    parser = argparse.ArgumentParser(
        description="Crab video detection is disabled while video sampling is rebuilt on top of the image detector.",
    )
    parser.add_argument("video", nargs="?", help="Video path reserved for the future detector.")
    parser.add_argument("--output-dir", default="", help=argparse.SUPPRESS)
    parser.add_argument("--interval", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--max-count", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--sample-step", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--start", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--end", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--no-force-square", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--unwrap-size", nargs=2, type=int, default=None, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Report that batch crab video detection is intentionally unavailable."""
    build_parser().parse_args(argv)
    print("Crab video detection is currently disabled while video sampling is rebuilt on top of the image detector.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
